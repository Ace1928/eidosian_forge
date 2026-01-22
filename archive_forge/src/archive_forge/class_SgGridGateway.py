from __future__ import absolute_import, division, print_function
import ansible_collections.netapp.storagegrid.plugins.module_utils.netapp as netapp_utils
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.storagegrid.plugins.module_utils.netapp import SGRestAPI
class SgGridGateway:
    """
    Create, modify and delete Gateway entries for StorageGRID
    """

    def __init__(self):
        """
        Parse arguments, setup state variables,
        check parameters and ensure request module is installed
        """
        self.argument_spec = netapp_utils.na_storagegrid_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), gateway_id=dict(required=False, type='str'), display_name=dict(required=False, type='str'), port=dict(required=True, type='int'), secure=dict(required=False, type='bool', default=True), enable_ipv4=dict(required=False, type='bool', default=True), enable_ipv6=dict(required=False, type='bool', default=True), binding_mode=dict(required=False, type='str', choices=['global', 'ha-groups', 'node-interfaces'], default='global'), ha_groups=dict(required=False, type='list', elements='str'), node_interfaces=dict(required=False, type='list', elements='dict', options=dict(node=dict(required=False, type='str'), interface=dict(required=False, type='str'))), default_service_type=dict(required=False, type='str', choices=['s3', 'swift'], default='s3'), server_certificate=dict(required=False, type='str'), ca_bundle=dict(required=False, type='str'), private_key=dict(required=False, type='str', no_log=True)))
        parameter_map_gateway = {'gateway_id': 'id', 'display_name': 'displayName', 'port': 'port', 'secure': 'secure', 'enable_ipv4': 'enableIPv4', 'enable_ipv6': 'enableIPv6'}
        parameter_map_server = {'server_certificate': 'serverCertificateEncoded', 'ca_bundle': 'caBundleEncoded', 'private_key': 'privateKeyEncoded'}
        self.module = AnsibleModule(argument_spec=self.argument_spec, required_if=[('state', 'present', ['display_name'])], supports_check_mode=True)
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = SGRestAPI(self.module)
        self.rest_api.get_sg_product_version()
        self.data_gateway = {}
        self.data_gateway['accountId'] = '0'
        for k in parameter_map_gateway.keys():
            if self.parameters.get(k) is not None:
                self.data_gateway[parameter_map_gateway[k]] = self.parameters[k]
        self.data_server = {}
        self.data_server['defaultServiceType'] = self.parameters['default_service_type']
        if self.parameters['secure']:
            self.data_server['plaintextCertData'] = {}
            self.data_server['certSource'] = 'plaintext'
            for k in parameter_map_server.keys():
                if self.parameters.get(k) is not None:
                    self.data_server['plaintextCertData'][parameter_map_server[k]] = self.parameters[k]
        if self.parameters['binding_mode'] != 'global':
            self.rest_api.fail_if_not_sg_minimum_version('non-global binding mode', 11, 5)
        if self.parameters['binding_mode'] == 'ha-groups':
            self.data_gateway['pinTargets'] = {}
            self.data_gateway['pinTargets']['haGroups'] = self.build_ha_group_list()
            self.data_gateway['pinTargets']['nodeInterfaces'] = []
        elif self.parameters['binding_mode'] == 'node-interfaces':
            self.data_gateway['pinTargets'] = {}
            self.data_gateway['pinTargets']['nodeInterfaces'] = self.build_node_interface_list()
            self.data_gateway['pinTargets']['haGroups'] = []
        else:
            self.data_gateway['pinTargets'] = {}
            self.data_gateway['pinTargets']['haGroups'] = []
            self.data_gateway['pinTargets']['nodeInterfaces'] = []

    def build_ha_group_list(self):
        ha_group_ids = []
        api = 'api/v3/private/ha-groups'
        ha_groups, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        for param in self.parameters['ha_groups']:
            ha_group = next((item for item in ha_groups['data'] if item['name'] == param or item['id'] == param), None)
            if ha_group is not None:
                ha_group_ids.append(ha_group['id'])
            else:
                self.module.fail_json(msg="HA Group '%s' is invalid" % param)
        return ha_group_ids

    def build_node_interface_list(self):
        node_interfaces = []
        api = 'api/v3/grid/node-health'
        nodes, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        for node_interface in self.parameters['node_interfaces']:
            node_dict = {}
            node = next((item for item in nodes['data'] if item['name'] == node_interface['node']), None)
            if node is not None:
                node_dict['nodeId'] = node['id']
                node_dict['interface'] = node_interface['interface']
                node_interfaces.append(node_dict)
            else:
                self.module.fail_json(msg="Node '%s' is invalid" % node_interface['node'])
        return node_interfaces

    def get_grid_gateway_config(self, gateway_id):
        api = 'api/v3/private/gateway-configs/%s' % gateway_id
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        gateway = response['data']
        gateway_config = self.get_grid_gateway_server_config(gateway['id'])
        return (gateway, gateway_config)

    def get_grid_gateway_server_config(self, gateway_id):
        api = 'api/v3/private/gateway-configs/%s/server-config' % gateway_id
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def get_grid_gateway_ports(self, target_port):
        configured_ports = []
        gateway = {}
        gateway_config = {}
        api = 'api/v3/private/gateway-configs'
        response, error = self.rest_api.get(api)
        if error:
            self.module.fail_json(msg=error)
        grid_gateway_ports = response['data']
        configured_ports = [data['port'] for data in grid_gateway_ports]
        for index, port in enumerate(configured_ports):
            if target_port == port and grid_gateway_ports[index]['displayName'] == self.parameters['display_name']:
                gateway = grid_gateway_ports[index]
                gateway_config = self.get_grid_gateway_server_config(gateway['id'])
                break
        return (gateway, gateway_config)

    def create_grid_gateway(self):
        api = 'api/v3/private/gateway-configs'
        response, error = self.rest_api.post(api, self.data_gateway)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def delete_grid_gateway(self, gateway_id):
        api = 'api/v3/private/gateway-configs/' + gateway_id
        self.data = None
        response, error = self.rest_api.delete(api, self.data)
        if error:
            self.module.fail_json(msg=error)

    def update_grid_gateway(self, gateway_id):
        api = 'api/v3/private/gateway-configs/%s' % gateway_id
        response, error = self.rest_api.put(api, self.data_gateway)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def update_grid_gateway_server(self, gateway_id):
        api = 'api/v3/private/gateway-configs/%s/server-config' % gateway_id
        response, error = self.rest_api.put(api, self.data_server)
        if error:
            self.module.fail_json(msg=error)
        return response['data']

    def apply(self):
        gateway = None
        gateway_config = None
        update_gateway = False
        update_gateway_server = False
        if self.parameters.get('gateway_id'):
            gateway, gateway_config = self.get_grid_gateway_config(self.parameters['gateway_id'])
        else:
            gateway, gateway_config = self.get_grid_gateway_ports(self.data_gateway['port'])
        cd_action = self.na_helper.get_cd_action(gateway.get('id'), self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            update = False
            if self.data_server.get('plaintextCertData'):
                if self.data_server['plaintextCertData'].get('privateKeyEncoded') is not None:
                    update = True
                    self.module.warn('This module is not idempotent when private_key is present.')
            if gateway_config.get('plaintextCertData'):
                if gateway_config['plaintextCertData'].get('metadata'):
                    del gateway_config['plaintextCertData']['metadata']
            if self.rest_api.meets_sg_minimum_version(11, 5):
                update_gateway = self.na_helper.get_modified_attributes(gateway, self.data_gateway)
            update_gateway_server = self.na_helper.get_modified_attributes(gateway_config, self.data_server)
            if update:
                self.na_helper.changed = True
        result_message = ''
        resp_data = {}
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'delete':
                self.delete_grid_gateway(gateway['id'])
                result_message = 'Load Balancer Gateway Port Deleted'
            elif cd_action == 'create':
                resp_data = self.create_grid_gateway()
                gateway['id'] = resp_data['id']
                resp_data_server = self.update_grid_gateway_server(gateway['id'])
                resp_data.update(resp_data_server)
                result_message = 'Load Balancer Gateway Port Created'
            else:
                resp_data = gateway
                if update_gateway:
                    resp_data = self.update_grid_gateway(gateway['id'])
                    resp_data.update(gateway_config)
                if update_gateway_server:
                    resp_data_server = self.update_grid_gateway_server(gateway['id'])
                    resp_data.update(resp_data_server)
                result_message = 'Load Balancer Gateway Port Updated'
        self.module.exit_json(changed=self.na_helper.changed, msg=result_message, resp=resp_data)