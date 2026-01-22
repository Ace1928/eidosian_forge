from __future__ import absolute_import, division, print_function
import json
import logging
from pprint import pformat
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
class IscsiInterface(object):

    def __init__(self):
        argument_spec = eseries_host_argument_spec()
        argument_spec.update(dict(controller=dict(type='str', required=True, choices=['A', 'B']), name=dict(type='int', aliases=['channel']), state=dict(type='str', required=False, default='enabled', choices=['enabled', 'disabled']), address=dict(type='str', required=False), subnet_mask=dict(type='str', required=False), gateway=dict(type='str', required=False), config_method=dict(type='str', required=False, default='dhcp', choices=['dhcp', 'static']), mtu=dict(type='int', default=1500, required=False, aliases=['max_frame_size']), log_path=dict(type='str', required=False)))
        required_if = [['config_method', 'static', ['address', 'subnet_mask']]]
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True, required_if=required_if)
        args = self.module.params
        self.controller = args['controller']
        self.name = args['name']
        self.mtu = args['mtu']
        self.state = args['state']
        self.address = args['address']
        self.subnet_mask = args['subnet_mask']
        self.gateway = args['gateway']
        self.config_method = args['config_method']
        self.ssid = args['ssid']
        self.url = args['api_url']
        self.creds = dict(url_password=args['api_password'], validate_certs=args['validate_certs'], url_username=args['api_username'])
        self.check_mode = self.module.check_mode
        self.post_body = dict()
        self.controllers = list()
        log_path = args['log_path']
        self._logger = logging.getLogger(self.__class__.__name__)
        if log_path:
            logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode='w', format='%(relativeCreated)dms %(levelname)s %(module)s.%(funcName)s:%(lineno)d\n %(message)s')
        if not self.url.endswith('/'):
            self.url += '/'
        if self.mtu < 1500 or self.mtu > 9000:
            self.module.fail_json(msg='The provided mtu is invalid, it must be > 1500 and < 9000 bytes.')
        if self.config_method == 'dhcp' and any([self.address, self.subnet_mask, self.gateway]):
            self.module.fail_json(msg='A config_method of dhcp is mutually exclusive with the address, subnet_mask, and gateway options.')
        address_regex = re.compile('\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}\\.\\d{1,3}')
        if self.address and (not address_regex.match(self.address)):
            self.module.fail_json(msg='An invalid ip address was provided for address.')
        if self.subnet_mask and (not address_regex.match(self.subnet_mask)):
            self.module.fail_json(msg='An invalid ip address was provided for subnet_mask.')
        if self.gateway and (not address_regex.match(self.gateway)):
            self.module.fail_json(msg='An invalid ip address was provided for gateway.')

    @property
    def interfaces(self):
        ifaces = list()
        try:
            rc, ifaces = request(self.url + 'storage-systems/%s/graph/xpath-filter?query=/controller/hostInterfaces' % self.ssid, headers=HEADERS, **self.creds)
        except Exception as err:
            self.module.fail_json(msg='Failed to retrieve defined host interfaces. Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        ifaces = [iface['iscsi'] for iface in ifaces if iface['interfaceType'] == 'iscsi']
        return ifaces

    def get_controllers(self):
        """Retrieve a mapping of controller labels to their references
        {
            'A': '070000000000000000000001',
            'B': '070000000000000000000002',
        }
        :return: the controllers defined on the system
        """
        controllers = list()
        try:
            rc, controllers = request(self.url + 'storage-systems/%s/graph/xpath-filter?query=/controller/id' % self.ssid, headers=HEADERS, **self.creds)
        except Exception as err:
            self.module.fail_json(msg='Failed to retrieve controller list! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        controllers.sort()
        controllers_dict = {}
        i = ord('A')
        for controller in controllers:
            label = chr(i)
            controllers_dict[label] = controller
            i += 1
        return controllers_dict

    def fetch_target_interface(self):
        interfaces = self.interfaces
        for iface in interfaces:
            if iface['channel'] == self.name and self.controllers[self.controller] == iface['controllerId']:
                return iface
        channels = sorted(set((str(iface['channel']) for iface in interfaces if self.controllers[self.controller] == iface['controllerId'])))
        self.module.fail_json(msg='The requested channel of %s is not valid. Valid channels include: %s.' % (self.name, ', '.join(channels)))

    def make_update_body(self, target_iface):
        body = dict(iscsiInterface=target_iface['id'])
        update_required = False
        self._logger.info('Requested state=%s.', self.state)
        self._logger.info('config_method: current=%s, requested=%s', target_iface['ipv4Data']['ipv4AddressConfigMethod'], self.config_method)
        if self.state == 'enabled':
            settings = dict()
            if not target_iface['ipv4Enabled']:
                update_required = True
                settings['ipv4Enabled'] = [True]
            if self.mtu != target_iface['interfaceData']['ethernetData']['maximumFramePayloadSize']:
                update_required = True
                settings['maximumFramePayloadSize'] = [self.mtu]
            if self.config_method == 'static':
                ipv4Data = target_iface['ipv4Data']['ipv4AddressData']
                if ipv4Data['ipv4Address'] != self.address:
                    update_required = True
                    settings['ipv4Address'] = [self.address]
                if ipv4Data['ipv4SubnetMask'] != self.subnet_mask:
                    update_required = True
                    settings['ipv4SubnetMask'] = [self.subnet_mask]
                if self.gateway is not None and ipv4Data['ipv4GatewayAddress'] != self.gateway:
                    update_required = True
                    settings['ipv4GatewayAddress'] = [self.gateway]
                if target_iface['ipv4Data']['ipv4AddressConfigMethod'] != 'configStatic':
                    update_required = True
                    settings['ipv4AddressConfigMethod'] = ['configStatic']
            elif target_iface['ipv4Data']['ipv4AddressConfigMethod'] != 'configDhcp':
                update_required = True
                settings.update(dict(ipv4Enabled=[True], ipv4AddressConfigMethod=['configDhcp']))
            body['settings'] = settings
        elif target_iface['ipv4Enabled']:
            update_required = True
            body['settings'] = dict(ipv4Enabled=[False])
        self._logger.info('Update required ?=%s', update_required)
        self._logger.info('Update body: %s', pformat(body))
        return (update_required, body)

    def update(self):
        self.controllers = self.get_controllers()
        if self.controller not in self.controllers:
            self.module.fail_json(msg='The provided controller name is invalid. Valid controllers: %s.' % ', '.join(self.controllers.keys()))
        iface_before = self.fetch_target_interface()
        update_required, body = self.make_update_body(iface_before)
        if update_required and (not self.check_mode):
            try:
                url = self.url + 'storage-systems/%s/symbol/setIscsiInterfaceProperties' % self.ssid
                rc, result = request(url, method='POST', data=json.dumps(body), headers=HEADERS, timeout=300, ignore_errors=True, **self.creds)
                if rc == 422 and result['retcode'] in ['busy', '3']:
                    self.module.fail_json(msg='The interface is currently busy (probably processing a previously requested modification request). This operation cannot currently be completed. Array Id [%s]. Error [%s].' % (self.ssid, result))
                elif rc != 200:
                    self.module.fail_json(msg='Failed to modify the interface! Array Id [%s]. Error [%s].' % (self.ssid, to_native(result)))
                self._logger.debug('Update request completed successfully.')
            except Exception as err:
                self.module.fail_json(msg='Connection failure: we failed to modify the interface! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        iface_after = self.fetch_target_interface()
        self.module.exit_json(msg='The interface settings have been updated.', changed=update_required, enabled=iface_after['ipv4Enabled'])

    def __call__(self, *args, **kwargs):
        self.update()