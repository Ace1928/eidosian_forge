from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, netapp_ipaddress
class NetAppOntapBgpPeerGroup:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), name=dict(required=True, type='str'), from_name=dict(required=False, type='str'), ipspace=dict(required=False, type='str'), local=dict(required=False, type='dict', options=dict(interface=dict(required=False, type='dict', options=dict(name=dict(required=False, type='str'))), ip=dict(required=False, type='dict', options=dict(address=dict(required=False, type='str'), netmask=dict(required=False, type='str'))), port=dict(required=False, type='dict', options=dict(name=dict(required=False, type='str'), node=dict(required=False, type='dict', options=dict(name=dict(required=False, type='str'))))))), peer=dict(required=False, type='dict', options=dict(address=dict(required=False, type='str'), asn=dict(required=False, type='int')))))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.uuid = None
        self.na_helper = NetAppModule(self.module)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        if self.na_helper.safe_get(self.parameters, ['peer', 'address']):
            self.parameters['peer']['address'] = netapp_ipaddress.validate_and_compress_ip_address(self.parameters['peer']['address'], self.module)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_bgp_peer_group', 9, 7)
        self.parameters = self.na_helper.filter_out_none_entries(self.parameters)

    def get_bgp_peer_group(self, name=None):
        """
        Get BGP peer group.
        """
        if name is None:
            name = self.parameters['name']
        api = 'network/ip/bgp/peer-groups'
        query = {'name': name, 'fields': 'name,uuid,peer'}
        if 'ipspace' in self.parameters:
            query['ipspace.name'] = self.parameters['ipspace']
        record, error = rest_generic.get_one_record(self.rest_api, api, query)
        if error:
            self.module.fail_json(msg='Error fetching BGP peer group %s: %s' % (name, to_native(error)), exception=traceback.format_exc())
        if record:
            self.uuid = record['uuid']
            return {'name': self.na_helper.safe_get(record, ['name']), 'peer': self.na_helper.safe_get(record, ['peer'])}
        return None

    def create_bgp_peer_group(self):
        """
        Create BGP peer group.
        """
        api = 'network/ip/bgp/peer-groups'
        body = {'name': self.parameters['name'], 'local': self.parameters['local'], 'peer': self.parameters['peer']}
        if 'ipspace' in self.parameters:
            body['ipspace.name'] = self.parameters['ipspace']
        dummy, error = rest_generic.post_async(self.rest_api, api, body)
        if error:
            self.module.fail_json(msg='Error creating BGP peer group %s: %s.' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def modify_bgp_peer_group(self, modify):
        """
        Modify BGP peer group.
        """
        api = 'network/ip/bgp/peer-groups'
        body = {}
        if 'name' in modify:
            body['name'] = modify['name']
        if 'peer' in modify:
            body['peer'] = modify['peer']
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body)
        if error:
            name = self.parameters['from_name'] if 'name' in modify else self.parameters['name']
            self.module.fail_json(msg='Error modifying BGP peer group %s: %s.' % (name, to_native(error)), exception=traceback.format_exc())

    def delete_bgp_peer_group(self):
        """
        Delete BGP peer group.
        """
        api = 'network/ip/bgp/peer-groups'
        dummy, error = rest_generic.delete_async(self.rest_api, api, self.uuid)
        if error:
            self.module.fail_json(msg='Error deleting BGP peer group %s: %s.' % (self.parameters['name'], to_native(error)), exception=traceback.format_exc())

    def apply(self):
        current = self.get_bgp_peer_group()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        modify = None
        if cd_action == 'create':
            if self.parameters.get('from_name'):
                current = self.get_bgp_peer_group(self.parameters['from_name'])
                if not current:
                    self.module.fail_json(msg='Error renaming BGP peer group, %s does not exist.' % self.parameters['from_name'])
                cd_action = None
            elif not self.parameters.get('local') or not self.parameters.get('peer'):
                self.module.fail_json(msg='Error creating BGP peer group %s, local and peer are required in create.' % self.parameters['name'])
        if cd_action is None:
            modify = self.na_helper.get_modified_attributes(current, self.parameters)
            if self.na_helper.safe_get(modify, ['peer', 'asn']):
                self.module.fail_json(msg='Error: cannot modify peer asn.')
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_bgp_peer_group()
            elif cd_action == 'delete':
                self.delete_bgp_peer_group()
            else:
                self.modify_bgp_peer_group(modify)
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)