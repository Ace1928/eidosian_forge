from __future__ import absolute_import, division, print_function
import json
import logging
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import request, eseries_host_argument_spec
from ansible.module_utils._text import to_native
class GlobalSettings(object):

    def __init__(self):
        argument_spec = eseries_host_argument_spec()
        argument_spec.update(dict(name=dict(type='str', required=False, aliases=['label']), log_path=dict(type='str', required=False)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        args = self.module.params
        self.name = args['name']
        self.ssid = args['ssid']
        self.url = args['api_url']
        self.creds = dict(url_password=args['api_password'], validate_certs=args['validate_certs'], url_username=args['api_username'])
        self.check_mode = self.module.check_mode
        log_path = args['log_path']
        self._logger = logging.getLogger(self.__class__.__name__)
        if log_path:
            logging.basicConfig(level=logging.DEBUG, filename=log_path, filemode='w', format='%(relativeCreated)dms %(levelname)s %(module)s.%(funcName)s:%(lineno)d\n %(message)s')
        if not self.url.endswith('/'):
            self.url += '/'
        if self.name and len(self.name) > 30:
            self.module.fail_json(msg='The provided name is invalid, it must be < 30 characters in length.')

    def get_name(self):
        try:
            rc, result = request(self.url + 'storage-systems/%s' % self.ssid, headers=HEADERS, **self.creds)
            if result['status'] in ['offline', 'neverContacted']:
                self.module.fail_json(msg='This storage-system is offline! Array Id [%s].' % self.ssid)
            return result['name']
        except Exception as err:
            self.module.fail_json(msg='Connection failure! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))

    def update_name(self):
        name = self.get_name()
        update = False
        if self.name != name:
            update = True
        body = dict(name=self.name)
        if update and (not self.check_mode):
            try:
                rc, result = request(self.url + 'storage-systems/%s/configuration' % self.ssid, method='POST', data=json.dumps(body), headers=HEADERS, **self.creds)
                self._logger.info('Set name to %s.', result['name'])
            except Exception as err:
                self.module.fail_json(msg='We failed to set the storage-system name! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return update

    def update(self):
        update = self.update_name()
        name = self.get_name()
        self.module.exit_json(msg='The requested settings have been updated.', changed=update, name=name)

    def __call__(self, *args, **kwargs):
        self.update()