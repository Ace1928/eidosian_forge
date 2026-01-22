from __future__ import (absolute_import, division, print_function)
import os
import copy
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import iteritems
class OpenshiftLDAPAugmentedActiveDirectory(OpenshiftLDAPRFC2307):

    def __init__(self, config, ldap_connection):
        self.config = config
        self.ldap_interface = self.create_ldap_interface(ldap_connection)

    def create_ldap_interface(self, connection):
        segment = self.config.get('augmentedActiveDirectory')
        user_base_query = openshift_ldap_build_base_query(segment['usersQuery'])
        groups_base_qry = openshift_ldap_build_base_query(segment['groupsQuery'])
        user_query = OpenshiftLDAPQuery(user_base_query)
        groups_query = OpenshiftLDAPQueryOnAttribute(groups_base_qry, segment['groupUIDAttribute'])
        return OpenshiftLDAP_AugmentedADInterface(connection=connection, user_query=user_query, group_member_attr=segment['groupMembershipAttributes'], user_name_attr=segment['userNameAttributes'], group_qry=groups_query, group_name_attr=segment['groupNameAttributes'])

    def is_ldapgroup_exists(self, uid):
        return self.ldap_interface.exists(uid)