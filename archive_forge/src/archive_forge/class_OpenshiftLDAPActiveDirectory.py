from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
class OpenshiftLDAPActiveDirectory(object):

    def __init__(self, config, ldap_connection):
        self.config = config
        self.ldap_interface = self.create_ldap_interface(ldap_connection)

    def create_ldap_interface(self, connection):
        segment = self.config.get('activeDirectory')
        base_query = openshift_ldap_build_base_query(segment['usersQuery'])
        user_query = OpenshiftLDAPQuery(base_query)
        return OpenshiftLDAP_ADInterface(connection=connection, user_query=user_query, group_member_attr=segment['groupMembershipAttributes'], user_name_attr=segment['userNameAttributes'])

    def get_username_for_entry(self, entry):
        username = openshift_ldap_get_attribute_for_entry(entry, self.ldap_interface.userNameAttributes)
        if not username:
            return (None, 'The user entry (%s) does not map to a OpenShift User name with the given mapping' % entry)
        return (username, None)

    def get_group_name_for_uid(self, uid):
        return (uid, None)

    def is_ldapgroup_exists(self, uid):
        members, error = self.extract_members(uid)
        if error:
            return (False, error)
        exists = members and len(members) > 0
        return (exists, None)

    def list_groups(self):
        return self.ldap_interface.list_groups()

    def extract_members(self, uid):
        return self.ldap_interface.extract_members(uid)