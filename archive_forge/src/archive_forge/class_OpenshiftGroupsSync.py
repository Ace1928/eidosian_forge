from __future__ import (absolute_import, division, print_function)
import traceback
from datetime import datetime
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import missing_required_lib
from ansible_collections.community.okd.plugins.module_utils.openshift_ldap import (
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
class OpenshiftGroupsSync(AnsibleOpenshiftModule):

    def __init__(self, **kwargs):
        super(OpenshiftGroupsSync, self).__init__(**kwargs)
        self.__k8s_group_api = None
        self.__ldap_connection = None
        self.host = None
        self.port = None
        self.netlocation = None
        self.scheme = None
        self.config = self.params.get('sync_config')
        if not HAS_PYTHON_LDAP:
            self.fail_json(msg=missing_required_lib('python-ldap'), error=to_native(PYTHON_LDAP_ERROR))

    @property
    def k8s_group_api(self):
        if not self.__k8s_group_api:
            params = dict(kind='Group', api_version='user.openshift.io/v1', fail=True)
            self.__k8s_group_api = self.find_resource(**params)
        return self.__k8s_group_api

    @property
    def hostIP(self):
        return self.netlocation

    @property
    def connection(self):
        if not self.__ldap_connection:
            params = dict(module=self, server_uri=self.config.get('url'), bind_dn=self.config.get('bindDN'), bind_pw=self.config.get('bindPassword'), insecure=boolean(self.config.get('insecure')), ca_file=self.config.get('ca'))
            self.__ldap_connection = connect_to_ldap(**params)
        return self.__ldap_connection

    def close_connection(self):
        if self.__ldap_connection:
            self.__ldap_connection.unbind_s()
        self.__ldap_connection = None

    def exit_json(self, **kwargs):
        self.close_connection()
        self.module.exit_json(**kwargs)

    def fail_json(self, **kwargs):
        self.close_connection()
        self.module.fail_json(**kwargs)

    def get_syncer(self):
        syncer = None
        if 'rfc2307' in self.config:
            syncer = OpenshiftLDAPRFC2307(self.config, self.connection)
        elif 'activeDirectory' in self.config:
            syncer = OpenshiftLDAPActiveDirectory(self.config, self.connection)
        elif 'augmentedActiveDirectory' in self.config:
            syncer = OpenshiftLDAPAugmentedActiveDirectory(self.config, self.connection)
        else:
            msg = "No schema-specific config was found, should be one of 'rfc2307', 'activeDirectory', 'augmentedActiveDirectory'"
            self.fail_json(msg=msg)
        return syncer

    def synchronize(self):
        sync_group_type = self.module.params.get('type')
        groups_uids = []
        ldap_openshift_group = OpenshiftLDAPGroups(module=self)
        syncer = self.get_syncer()
        if sync_group_type == 'openshift':
            groups_uids, err = ldap_openshift_group.list_groups()
            if err:
                self.fail_json(msg='Failed to list openshift groups', errors=err)
        else:
            groups_uids = self.params.get('allow_groups')
            if not groups_uids:
                groups_uids, err = syncer.list_groups()
                if err:
                    self.module.fail_json(msg=err)
            deny_groups = self.params.get('deny_groups')
            if deny_groups:
                groups_uids = [uid for uid in groups_uids if uid not in deny_groups]
        openshift_groups = []
        for uid in groups_uids:
            member_entries, err = syncer.extract_members(uid)
            if err:
                self.fail_json(msg=err)
            usernames = []
            for entry in member_entries:
                name, err = syncer.get_username_for_entry(entry)
                if err:
                    self.exit_json(msg='Unable to determine username for entry %s: %s' % (entry, err))
                if isinstance(name, list):
                    usernames.extend(name)
                else:
                    usernames.append(name)
            if sync_group_type == 'openshift':
                group_name, err = ldap_openshift_group.get_group_name_for_uid(uid)
            else:
                group_name, err = syncer.get_group_name_for_uid(uid)
            if err:
                self.exit_json(msg=err)
            group, err = ldap_openshift_group.make_openshift_group(uid, group_name, usernames)
            if err:
                self.fail_json(msg=err)
            openshift_groups.append(group)
        results, diffs, changed = ldap_openshift_group.create_openshift_groups(openshift_groups)
        self.module.exit_json(changed=True, groups=results)

    def prune(self):
        ldap_openshift_group = OpenshiftLDAPGroups(module=self)
        groups_uids, err = ldap_openshift_group.list_groups()
        if err:
            self.fail_json(msg='Failed to list openshift groups', errors=err)
        syncer = self.get_syncer()
        changed = False
        groups = []
        for uid in groups_uids:
            exists, err = syncer.is_ldapgroup_exists(uid)
            if err:
                msg = 'Error determining LDAP group existence for group %s: %s' % (uid, err)
                self.module.fail_json(msg=msg)
            if exists:
                continue
            group_name, err = ldap_openshift_group.get_group_name_for_uid(uid)
            if err:
                self.module.fail_json(msg=err)
            result = ldap_openshift_group.delete_openshift_group(group_name)
            groups.append(result)
            changed = True
        self.exit_json(changed=changed, groups=groups)

    def execute_module(self):
        error = validate_ldap_sync_config(self.config)
        if error:
            self.fail_json(msg='Invalid LDAP Sync config: %s' % error)
        if self.config.get('url'):
            result, error = ldap_split_host_port(self.config.get('url'))
            if error:
                self.fail_json(msg="Failed to parse url='{0}': {1}".format(self.config.get('url'), error))
            self.netlocation, self.host, self.port = (result['netlocation'], result['host'], result['port'])
            self.scheme = result['scheme']
        if self.params.get('state') == 'present':
            self.synchronize()
        else:
            self.prune()