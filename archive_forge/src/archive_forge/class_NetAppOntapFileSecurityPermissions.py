from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
from ansible_collections.netapp.ontap.plugins.module_utils.netapp_module import NetAppModule
from ansible_collections.netapp.ontap.plugins.module_utils import rest_generic, rest_vserver
class NetAppOntapFileSecurityPermissions:

    def __init__(self):
        self.argument_spec = netapp_utils.na_ontap_host_argument_spec()
        self.argument_spec.update(dict(state=dict(required=False, type='str', choices=['present', 'absent'], default='present'), vserver=dict(required=True, type='str'), path=dict(required=True, type='str'), owner=dict(required=False, type='str'), control_flags=dict(required=False, type='str'), group=dict(required=False, type='str'), access_control=dict(required=False, type='str', choices=['file_directory', 'slag']), ignore_paths=dict(required=False, type='list', elements='str'), propagation_mode=dict(required=False, type='str', choices=['propagate', 'replace']), acls=dict(type='list', elements='dict', options=dict(access=dict(required=True, type='str', choices=['access_allow', 'access_deny', 'access_allowed_callback', 'access_denied_callback', 'access_allowed_callback_object', 'access_denied_callback_object', 'system_audit_callback', 'system_audit_callback_object', 'system_resource_attribute', 'system_scoped_policy_id', 'audit_failure', 'audit_success', 'audit_success_and_failure']), access_control=dict(required=False, type='str', choices=['file_directory', 'slag']), user=dict(required=True, type='str', aliases=['acl_user']), rights=dict(required=False, choices=['no_access', 'full_control', 'modify', 'read_and_execute', 'read', 'write'], type='str'), apply_to=dict(required=True, type='dict', options=dict(files=dict(required=False, type='bool', default=False), sub_folders=dict(required=False, type='bool', default=False), this_folder=dict(required=False, type='bool', default=False))), advanced_rights=dict(required=False, type='dict', options=dict(append_data=dict(required=False, type='bool'), delete=dict(required=False, type='bool'), delete_child=dict(required=False, type='bool'), execute_file=dict(required=False, type='bool'), full_control=dict(required=False, type='bool'), read_attr=dict(required=False, type='bool'), read_data=dict(required=False, type='bool'), read_ea=dict(required=False, type='bool'), read_perm=dict(required=False, type='bool'), write_attr=dict(required=False, type='bool'), write_data=dict(required=False, type='bool'), write_ea=dict(required=False, type='bool'), write_owner=dict(required=False, type='bool'), write_perm=dict(required=False, type='bool'))), ignore_paths=dict(required=False, type='list', elements='str'), propagation_mode=dict(required=False, type='str', choices=['propagate', 'replace']))), validate_changes=dict(required=False, type='str', choices=['ignore', 'warn', 'error'], default='error')))
        self.module = AnsibleModule(argument_spec=self.argument_spec, supports_check_mode=True)
        self.svm_uuid = None
        self.na_helper = NetAppModule(self)
        self.parameters = self.na_helper.check_and_set_parameters(self.module)
        self.rest_api = netapp_utils.OntapRestAPI(self.module)
        self.rest_api.fail_if_not_rest_minimum_version('na_ontap_file_security_permissions', 9, 9, 1)
        dummy, error = self.rest_api.is_rest(partially_supported_rest_properties=[['access_control', (9, 10, 1)], ['acls.access_control', (9, 10, 1)]], parameters=self.parameters)
        if error:
            self.module.fail_json(msg=error)
        self.parameters = self.na_helper.filter_out_none_entries(self.parameters)
        self.apply_to_keys = ['files', 'sub_folders', 'this_folder']
        self.post_acl_keys = ['access', 'advanced_rights', 'apply_to', 'rights', 'user']
        if self.parameters['state'] == 'present':
            self.validate_acls()

    def validate_acls(self):
        if 'acls' not in self.parameters:
            return
        self.parameters['acls'] = self.na_helper.filter_out_none_entries(self.parameters['acls'])
        for acl in self.parameters['acls']:
            if 'rights' in acl:
                if 'advanced_rights' in acl:
                    self.module.fail_json(msg="Error: suboptions 'rights' and 'advanced_rights' are mutually exclusive.")
                self.module.warn('This module is not idempotent when "rights" is used, make sure to use "advanced_rights".')
            if not any((self.na_helper.safe_get(acl, ['apply_to', key]) for key in self.apply_to_keys)):
                self.module.fail_json(msg='Error: at least one suboption must be true for apply_to.  Got: %s' % acl)
            self.match_acl_with_acls(acl, self.parameters['acls'])
        for option in ('access_control', 'ignore_paths', 'propagation_mode'):
            value = self.parameters.get(option)
            if value is not None:
                for acl in self.parameters['acls']:
                    if acl.get(option) not in (None, value):
                        self.module.fail_json(msg='Error: mismatch between top level value and ACL value for %s: %s vs %s' % (option, value, acl.get(option)))
                    acl[option] = value

    @staticmethod
    def url_encode(url_fragment):
        """
            replace special characters with URL encoding:
            %2F for /, %5C for backslash
        """
        return url_fragment.replace('/', '%2F').replace('\\', '%5C')

    def get_svm_uuid(self):
        self.svm_uuid, dummy = rest_vserver.get_vserver_uuid(self.rest_api, self.parameters['vserver'], self.module, True)

    def get_file_security_permissions(self):
        api = 'protocols/file-security/permissions/%s/%s' % (self.svm_uuid, self.url_encode(self.parameters['path']))
        fields = 'acls,control_flags,group,owner'
        record, error = rest_generic.get_one_record(self.rest_api, api, {'fields': fields})
        if error:
            if '655865' in error and self.parameters['state'] == 'absent':
                return None
            self.module.fail_json(msg='Error fetching file security permissions %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())
        return self.form_current(record) if record else None

    def form_current(self, record):
        current = {'group': self.na_helper.safe_get(record, ['group']), 'owner': self.na_helper.safe_get(record, ['owner']), 'control_flags': self.na_helper.safe_get(record, ['control_flags']), 'path': record['path']}
        acls = []

        def form_acl(acl):
            advanced_rights_keys = ['append_data', 'delete', 'delete_child', 'execute_file', 'full_control', 'read_attr', 'read_data', 'read_ea', 'read_perm', 'write_attr', 'write_data', 'write_ea', 'write_owner', 'write_perm']
            advanced_rights = {}
            apply_to = {}
            if 'advanced_rights' in acl:
                for key in advanced_rights_keys:
                    advanced_rights[key] = acl['advanced_rights'].get(key, False)
            if 'apply_to' in acl:
                for key in self.apply_to_keys:
                    apply_to[key] = acl['apply_to'].get(key, False)
            return {'advanced_rights': advanced_rights or None, 'apply_to': apply_to or None}
        for acl in record.get('acls', []):
            each_acl = {'access': self.na_helper.safe_get(acl, ['access']), 'access_control': self.na_helper.safe_get(acl, ['access_control']), 'inherited': self.na_helper.safe_get(acl, ['inherited']), 'rights': self.na_helper.safe_get(acl, ['rights']), 'user': self.na_helper.safe_get(acl, ['user'])}
            each_acl.update(form_acl(acl))
            acls.append(each_acl)
        current['acls'] = acls or None
        return current

    @staticmethod
    def has_acls(current):
        return bool(current and current.get('acls'))

    def set_option(self, body, option):
        if self.parameters.get(option) is not None:
            body[option] = self.parameters[option]
            return True
        return False

    def sanitize_acl_for_post(self, acl):
        """ some fields like access_control, propagation_mode are not accepted for POST operation """
        post_acl = dict(acl)
        for key in acl:
            if key not in self.post_acl_keys:
                post_acl.pop(key)
        return post_acl

    def sanitize_acls_for_post(self, acls):
        """ some fields like access_control, propagation_mode are not accepted for POST operation """
        return [self.sanitize_acl_for_post(acl) for acl in acls]

    def create_file_security_permissions(self):
        api = 'protocols/file-security/permissions/%s/%s' % (self.svm_uuid, self.url_encode(self.parameters['path']))
        body = {}
        for option in ('access_control', 'control_flags', 'group', 'owner', 'ignore_paths', 'propagation_mode'):
            self.set_option(body, option)
        body['acls'] = self.sanitize_acls_for_post(self.parameters.get('acls', []))
        dummy, error = rest_generic.post_async(self.rest_api, api, body, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error creating file security permissions %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())

    def add_file_security_permissions_acl(self, acl):
        api = 'protocols/file-security/permissions/%s/%s/acl' % (self.svm_uuid, self.url_encode(self.parameters['path']))
        for option in ('access_control', 'propagation_mode'):
            self.set_option(acl, option)
        dummy, error = rest_generic.post_async(self.rest_api, api, acl, timeout=0)
        if error:
            self.module.fail_json(msg='Error adding file security permissions acl %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())

    def modify_file_security_permissions_acl(self, acl):
        api = 'protocols/file-security/permissions/%s/%s/acl' % (self.svm_uuid, self.url_encode(self.parameters['path']))
        acl = dict(acl)
        user = acl.pop('user')
        for option in ('access_control', 'propagation_mode'):
            self.set_option(acl, option)
        dummy, error = rest_generic.patch_async(self.rest_api, api, self.url_encode(user), acl, {'return_records': 'true'})
        if error:
            self.module.fail_json(msg='Error modifying file security permissions acl %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())

    def delete_file_security_permissions_acl(self, acl):
        api = 'protocols/file-security/permissions/%s/%s/acl' % (self.svm_uuid, self.url_encode(self.parameters['path']))
        acl = self.na_helper.filter_out_none_entries(acl)
        user = acl.pop('user')
        acl.pop('advanced_rights', None)
        acl.pop('rights', None)
        acl.pop('inherited', None)
        for option in ('access_control', 'propagation_mode'):
            self.set_option(acl, option)
        dummy, error = rest_generic.delete_async(self.rest_api, api, self.url_encode(user), {'return_records': 'true'}, acl, timeout=0)
        if error:
            self.module.fail_json(msg='Error deleting file security permissions acl %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())

    def modify_file_security_permissions(self, modify):
        api = 'protocols/file-security/permissions/%s/%s' % (self.svm_uuid, self.url_encode(self.parameters['path']))
        body = {}
        for option in modify:
            self.set_option(body, option)
        dummy, error = rest_generic.patch_async(self.rest_api, api, None, body, job_timeout=120)
        if error:
            self.module.fail_json(msg='Error modifying file security permissions %s: %s' % (self.parameters['path'], to_native(error)), exception=traceback.format_exc())

    def match_acl_with_acls(self, acl, acls):
        """ return acl if user and access and apply_to are matched, otherwiese None """
        matches = []
        for an_acl in acls:
            inherited = an_acl['inherited'] if 'inherited' in an_acl else False and (acl['inherited'] if 'inherited' in acl else False)
            if acl['user'] == an_acl['user'] and acl['access'] == an_acl['access'] and (acl.get('access_control', 'file_directory') == an_acl.get('access_control', 'file_directory')) and (acl['apply_to'] == an_acl['apply_to']) and (not inherited):
                matches.append(an_acl)
        if len(matches) > 1:
            self.module.fail_json(msg='Error: found more than one desired ACLs with same user, access, access_control and apply_to  %s' % matches)
        return matches[0] if matches else None

    def get_acl_actions_on_modify(self, modify, current):
        acl_actions = {'patch-acls': [], 'post-acls': [], 'delete-acls': []}
        if not self.has_acls(current):
            acl_actions['post-acls'] = modify['acls']
            return acl_actions
        for acl in modify['acls']:
            current_acl = self.match_acl_with_acls(acl, current['acls'])
            if current_acl:
                if self.is_modify_acl_required(acl, current_acl):
                    acl_actions['patch-acls'].append(acl)
            else:
                acl_actions['post-acls'].append(acl)
        for acl in current['acls']:
            desired_acl = self.match_acl_with_acls(acl, self.parameters['acls'])
            if not desired_acl and (not acl.get('inherited')) and (self.parameters.get('access_control') in (None, acl.get('access_control'))):
                acl_actions['delete-acls'].append(acl)
        return acl_actions

    def is_modify_acl_required(self, desired_acl, current_acl):
        current_acl_copy = current_acl.copy()
        current_acl_copy.pop('user')
        modify = self.na_helper.get_modified_attributes(current_acl_copy, desired_acl)
        return bool(modify)

    def get_acl_actions_on_delete(self, current):
        acl_actions = {'patch-acls': [], 'post-acls': [], 'delete-acls': []}
        self.na_helper.changed = False
        if current.get('acls'):
            for acl in current['acls']:
                if not acl.get('inherited') and self.parameters.get('access_control') in (None, acl.get('access_control')):
                    self.na_helper.changed = True
                    acl_actions['delete-acls'].append(acl)
        return acl_actions

    def get_modify_actions(self, current):
        modify = self.na_helper.get_modified_attributes(current, self.parameters)
        if 'path' in modify:
            self.module.fail_json(msg='Error: mismatch on path values: desired: %s, received: %s' % (self.parameters['path'], current['path']))
        if 'acls' in modify:
            acl_actions = self.get_acl_actions_on_modify(modify, current)
            del modify['acls']
        else:
            acl_actions = {'patch-acls': [], 'post-acls': [], 'delete-acls': []}
        if not any((acl_actions['patch-acls'], acl_actions['post-acls'], acl_actions['delete-acls'], modify)):
            self.na_helper.changed = False
        return (modify, acl_actions)

    def get_acl_actions_on_create(self):
        """
        POST does not accept access_control and propagation_mode at the ACL level, these are global values for all ACLs.
        Since the user could have a list of ACLs with mixed property we should useP OST the create the SD and 1 group of ACLs
        then loop over the remaining ACLS.
        """
        acls_groups = {}
        preferred_group = (None, None)
        special_accesses = []
        for acl in self.parameters.get('acls', []):
            access_control = acl.get('access_control', 'file_directory')
            propagation_mode = acl.get('propagation_mode', 'propagate')
            if access_control not in acls_groups:
                acls_groups[access_control] = {}
            if propagation_mode not in acls_groups[access_control]:
                acls_groups[access_control][propagation_mode] = []
            acls_groups[access_control][propagation_mode].append(acl)
            access = acl.get('access')
            if access not in ('access_allow', 'access_deny', 'audit_success', 'audit_failure'):
                if preferred_group == (None, None):
                    preferred_group = (access_control, propagation_mode)
                if preferred_group != (access_control, propagation_mode):
                    self.module.fail_json(msg='Error: acl %s with access %s conflicts with other ACLs using accesses: %s with different access_control or propagation_mode: %s.' % (acl, access, special_accesses, preferred_group))
                special_accesses.append(access)
        if preferred_group == (None, None):
            for acc_key, acc_value in sorted(acls_groups.items()):
                for prop_key, prop_value in sorted(acc_value.items()):
                    if prop_value:
                        preferred_group = (acc_key, prop_key)
                        break
                if preferred_group != (None, None):
                    break
        create_acls = []
        acl_actions = {'patch-acls': [], 'post-acls': [], 'delete-acls': []}
        for acc_key, acc_value in sorted(acls_groups.items()):
            for prop_key, prop_value in sorted(acc_value.items()):
                if (acc_key, prop_key) == preferred_group:
                    create_acls = prop_value
                    self.parameters['access_control'] = acc_key
                    self.parameters['propagation_mode'] = prop_key
                elif prop_value:
                    acl_actions['post-acls'].extend(prop_value)
        self.parameters['acls'] = create_acls
        return acl_actions

    def get_actions(self):
        current = self.get_file_security_permissions()
        cd_action = self.na_helper.get_cd_action(current, self.parameters)
        modify, acl_actions = self.get_modify_actions(current) if cd_action is None else (None, {})
        if cd_action == 'create' and self.parameters.get('access_control') is None:
            acl_actions = self.get_acl_actions_on_create()
        if cd_action == 'delete':
            acl_actions = self.get_acl_actions_on_delete(current)
            cd_action = None
        return (cd_action, modify, acl_actions)

    def apply(self):
        self.get_svm_uuid()
        cd_action, modify, acl_actions = self.get_actions()
        if self.na_helper.changed and (not self.module.check_mode):
            if cd_action == 'create':
                self.create_file_security_permissions()
            if modify:
                self.modify_file_security_permissions(modify)
            for delete_acl in acl_actions.get('delete-acls', []):
                self.delete_file_security_permissions_acl(delete_acl)
            for patch_acl in acl_actions.get('patch-acls', []):
                self.modify_file_security_permissions_acl(patch_acl)
            for post_acl in acl_actions.get('post-acls', []):
                self.add_file_security_permissions_acl(post_acl)
            changed = self.na_helper.changed
            self.validate_changes(cd_action, modify)
            self.na_helper.changed = changed
        result = netapp_utils.generate_result(self.na_helper.changed, cd_action, modify)
        self.module.exit_json(**result)

    def validate_changes(self, cd_action, modify):
        if self.parameters['validate_changes'] == 'ignore':
            return
        new_cd_action, new_modify, acl_actions = self.get_actions()
        errors = []
        if new_cd_action is not None:
            errors.append('%s still required after %s (with modify: %s)' % (new_cd_action, cd_action, modify))
        if new_modify:
            errors.append('modify: %s still required after %s' % (new_modify, modify))
        errors.extend(('%s still required for %s' % (key, value) for key, value in acl_actions.items() if value))
        if errors:
            msg = 'Error - %s' % ' - '.join(errors)
            if self.parameters['validate_changes'] == 'error':
                self.module.fail_json(msg=msg)
            if self.parameters['validate_changes'] == 'warn':
                self.module.warn(msg)