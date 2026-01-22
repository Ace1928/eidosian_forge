from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils.netapp import OntapRestAPI
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic
import ansible_collections.netapp.ontap.plugins.module_utils.rest_response_helpers as rrh
class NetAppONTAPVolumeClone:
    """
        Creates a volume clone
    """

    def __init__(self):
        """
            Initialize the NetAppOntapVolumeClone class
        """
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present'], default='present'), parent_volume=dict(required=True, type='str'), name=dict(required=True, type='str', aliases=['volume']), vserver=dict(required=True, type='str'), parent_snapshot=dict(required=False, type='str', default=None), parent_vserver=dict(required=False, type='str', default=None), qos_policy_group_name=dict(required=False, type='str', default=None), space_reserve=dict(required=False, type='str', choices=['volume', 'none'], default=None), volume_type=dict(required=False, type='str', choices=['rw', 'dp']), junction_path=dict(required=False, type='str', default=None), uid=dict(required=False, type='int'), gid=dict(required=False, type='int'), split=dict(required=False, type='bool', default=None)))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True, required_together=[['uid', 'gid']], mutually_exclusive=[('junction_path', 'parent_vserver'), ('uid', 'parent_vserver'), ('gid', 'parent_vserver')])
        self.uuid = None
        self.na_helper = NetAppModule()
        self.parameters = self.na_helper.set_parameters(self.module.params)
        self.rest_api = OntapRestAPI(self.module)
        unsupported_rest_properties = ['space_reserve']
        self.use_rest = self.rest_api.is_rest_supported_properties(self.parameters, unsupported_rest_properties)
        if not self.use_rest:
            if not netapp_utils.has_netapp_lib():
                self.module.fail_json(msg=netapp_utils.netapp_lib_is_required())
            if self.parameters.get('parent_vserver'):
                self.create_server = netapp_utils.setup_na_ontap_zapi(module=self.module)
                self.vserver = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])
            else:
                self.vserver = netapp_utils.setup_na_ontap_zapi(module=self.module, vserver=self.parameters['vserver'])
                self.create_server = self.vserver

    def create_volume_clone(self):
        """
        Creates a new volume clone
        """
        if self.use_rest:
            return self.create_volume_clone_rest()
        clone_obj = netapp_utils.zapi.NaElement('volume-clone-create')
        clone_obj.add_new_child('parent-volume', self.parameters['parent_volume'])
        clone_obj.add_new_child('volume', self.parameters['name'])
        if self.parameters.get('qos_policy_group_name'):
            clone_obj.add_new_child('qos-policy-group-name', self.parameters['qos_policy_group_name'])
        if self.parameters.get('space_reserve'):
            clone_obj.add_new_child('space-reserve', self.parameters['space_reserve'])
        if self.parameters.get('parent_snapshot'):
            clone_obj.add_new_child('parent-snapshot', self.parameters['parent_snapshot'])
        if self.parameters.get('parent_vserver'):
            clone_obj.add_new_child('parent-vserver', self.parameters['parent_vserver'])
            clone_obj.add_new_child('vserver', self.parameters['vserver'])
        if self.parameters.get('volume_type'):
            clone_obj.add_new_child('volume-type', self.parameters['volume_type'])
        if self.parameters.get('junction_path'):
            clone_obj.add_new_child('junction-path', self.parameters['junction_path'])
        if self.parameters.get('uid'):
            clone_obj.add_new_child('uid', str(self.parameters['uid']))
            clone_obj.add_new_child('gid', str(self.parameters['gid']))
        try:
            self.create_server.invoke_successfully(clone_obj, True)
        except netapp_utils.zapi.NaApiError as exc:
            self.module.fail_json(msg='Error creating volume clone: %s: %s' % (self.parameters['name'], to_native(exc)))

    def modify_volume_clone(self):
        """
        Modify an existing volume clone
        """
        if 'split' in self.parameters and self.parameters['split']:
            self.start_volume_clone_split()

    def start_volume_clone_split(self):
        """
        Starts a volume clone split
        """
        if self.use_rest:
            return self.start_volume_clone_split_rest()
        clone_obj = netapp_utils.zapi.NaElement('volume-clone-split-start')
        clone_obj.add_new_child('volume', self.parameters['name'])
        try:
            self.vserver.invoke_successfully(clone_obj, True)
        except netapp_utils.zapi.NaApiError as exc:
            self.module.fail_json(msg='Error starting volume clone split: %s: %s' % (self.parameters['name'], to_native(exc)))

    def get_volume_clone(self):
        if self.use_rest:
            return self.get_volume_clone_rest()
        clone_obj = netapp_utils.zapi.NaElement('volume-clone-get')
        clone_obj.add_new_child('volume', self.parameters['name'])
        try:
            results = self.vserver.invoke_successfully(clone_obj, True)
        except netapp_utils.zapi.NaApiError as error:
            if to_native(error.code) == '15661':
                return None
            self.module.fail_json(msg='Error fetching volume clone information %s: %s' % (self.parameters['name'], to_native(error)))
        current = None
        if results.get_child_by_name('attributes'):
            attributes = results.get_child_by_name('attributes')
            info = attributes.get_child_by_name('volume-clone-info')
            current = {'split': bool(info.get_child_by_name('block-percentage-complete') or info.get_child_by_name('blocks-scanned') or info.get_child_by_name('blocks-updated'))}
        return current

    def get_volume_clone_rest(self):
        api = 'storage/volumes'
        params = {'name': self.parameters['name'], 'svm.name': self.parameters['vserver'], 'fields': 'clone.is_flexclone,uuid'}
        record, error = rest_generic.get_one_record(self.rest_api, api, params)
        if error:
            self.module.fail_json(msg='Error getting volume clone %s: %s' % (self.parameters['name'], to_native(error)))
        if record:
            return self.format_get_volume_clone_rest(record)
        return record

    def format_get_volume_clone_rest(self, record):
        return {'name': record.get('name', None), 'uuid': record.get('uuid', None), 'is_clone': self.na_helper.safe_get(record, ['clone', 'is_flexclone']), 'split': self.na_helper.safe_get(record, ['clone', 'is_flexclone']) is not True}

    def create_volume_clone_rest(self):
        api = 'storage/volumes'
        body = {'name': self.parameters['name'], 'clone.parent_volume.name': self.parameters['parent_volume'], 'clone.is_flexclone': True, 'svm.name': self.parameters['vserver']}
        if self.parameters.get('qos_policy_group_name'):
            body['qos.policy.name'] = self.parameters['qos_policy_group_name']
        if self.parameters.get('parent_snapshot'):
            body['clone.parent_snapshot.name'] = self.parameters['parent_snapshot']
        if self.parameters.get('parent_vserver'):
            body['clone.parent_svm.name'] = self.parameters['parent_vserver']
        if self.parameters.get('volume_type'):
            body['type'] = self.parameters['volume_type']
        if self.parameters.get('junction_path'):
            body['nas.path'] = self.parameters['junction_path']
        if self.parameters.get('uid'):
            body['nas.uid'] = self.parameters['uid']
        if self.parameters.get('gid'):
            body['nas.gid'] = self.parameters['gid']
        query = {'return_records': 'true'}
        response, error = rest_generic.post_async(self.rest_api, api, body, query, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error creating volume clone %s: %s' % (self.parameters['name'], to_native(error)))
        if response:
            record, error = rrh.check_for_0_or_1_records(api, response, error, query)
            if not error and record and ('uuid' not in record):
                error = 'uuid key not present in %s:' % record
            if error:
                self.module.fail_json(msg='Error: failed to parse create clone response: %s' % error)
            if record:
                self.uuid = record['uuid']

    def start_volume_clone_split_rest(self):
        if self.uuid is None:
            self.module.fail_json(msg='Error starting volume clone split %s: %s' % (self.parameters['name'], 'clone UUID is not set'))
        api = 'storage/volumes'
        body = {'clone.split_initiated': True}
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.uuid, body, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error starting volume clone split %s: %s' % (self.parameters['name'], to_native(error)))

    def apply(self):
        """
        Run Module based on playbook
        """
        current = self.get_volume_clone()
        if self.use_rest and current:
            self.uuid = current['uuid']
        if self.use_rest and current and (not current['is_clone']) and (not self.parameters.get('split')):
            self.module.fail_json(msg='Error: a volume %s which is not a FlexClone already exists, and split not requested.' % self.parameters['name'])
        modify = None
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        if cd_action is None and self.parameters['state'] == 'present':
            current_split = {'split': current.get('split')} if current else None
            modify = self.na_helper.get_modified_attributes(current_split, self.parameters)
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_volume_clone()
                if self.parameters.get('split'):
                    self.modify_volume_clone()
            if modify:
                self.modify_volume_clone()
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)