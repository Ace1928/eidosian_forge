import uuid
import ldap.filter
from oslo_log import log
from oslo_log import versionutils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone.identity.backends import base
from keystone.identity.backends.ldap import common as common_ldap
from keystone.identity.backends.ldap import models
class GroupApi(common_ldap.BaseLdap):
    DEFAULT_OU = 'ou=UserGroups'
    DEFAULT_STRUCTURAL_CLASSES = []
    DEFAULT_OBJECTCLASS = 'groupOfNames'
    DEFAULT_ID_ATTR = 'cn'
    DEFAULT_MEMBER_ATTRIBUTE = 'member'
    NotFound = exception.GroupNotFound
    options_name = 'group'
    attribute_options_names = {'description': 'desc', 'name': 'name'}
    immutable_attrs = ['name']
    model = models.Group

    def _ldap_res_to_model(self, res):
        model = super(GroupApi, self)._ldap_res_to_model(res)
        model['dn'] = res[0]
        return model

    def __init__(self, conf):
        super(GroupApi, self).__init__(conf)
        self.group_ad_nesting = conf.ldap.group_ad_nesting
        self.member_attribute = conf.ldap.group_member_attribute or self.DEFAULT_MEMBER_ATTRIBUTE

    def create(self, values):
        data = values.copy()
        if data.get('id') is None:
            data['id'] = uuid.uuid4().hex
        if 'description' in data and data['description'] in ['', None]:
            data.pop('description')
        return super(GroupApi, self).create(data)

    def update(self, group_id, values):
        old_obj = self.get(group_id)
        return super(GroupApi, self).update(group_id, values, old_obj)

    def add_user(self, user_dn, group_id, user_id):
        group_ref = self.get(group_id)
        group_dn = group_ref['dn']
        try:
            super(GroupApi, self).add_member(user_dn, group_dn)
        except exception.Conflict:
            raise exception.Conflict(_('User %(user_id)s is already a member of group %(group_id)s') % {'user_id': user_id, 'group_id': group_id})

    def list_user_groups(self, user_dn):
        """Return a list of groups for which the user is a member."""
        user_dn_esc = ldap.filter.escape_filter_chars(user_dn)
        if self.group_ad_nesting:
            query = '(%s:%s:=%s)' % (self.member_attribute, LDAP_MATCHING_RULE_IN_CHAIN, user_dn_esc)
        else:
            query = '(%s=%s)' % (self.member_attribute, user_dn_esc)
        return self.get_all(query)

    def list_user_groups_filtered(self, user_dn, hints):
        """Return a filtered list of groups for which the user is a member."""
        user_dn_esc = ldap.filter.escape_filter_chars(user_dn)
        if self.group_ad_nesting:
            query = '(member:%s:=%s)' % (LDAP_MATCHING_RULE_IN_CHAIN, user_dn_esc)
        else:
            query = '(%s=%s)' % (self.member_attribute, user_dn_esc)
        return self.get_all_filtered(hints, query)

    def list_group_users(self, group_id):
        """Return a list of user dns which are members of a group."""
        group_ref = self.get(group_id)
        group_dn = group_ref['dn']
        try:
            if self.group_ad_nesting:
                attrs = self._ldap_get_list(self.tree_dn, self.LDAP_SCOPE, query_params={'member:%s:' % LDAP_MATCHING_RULE_IN_CHAIN: group_dn}, attrlist=[self.member_attribute])
            else:
                attrs = self._ldap_get_list(group_dn, ldap.SCOPE_BASE, attrlist=[self.member_attribute])
        except ldap.NO_SUCH_OBJECT:
            raise self.NotFound(group_id=group_id)
        users = []
        for dn, member in attrs:
            user_dns = member.get(self.member_attribute, [])
            for user_dn in user_dns:
                users.append(user_dn)
        return users

    def get_filtered(self, group_id):
        group = self.get(group_id)
        return common_ldap.filter_entity(group)

    def get_filtered_by_name(self, group_name):
        group = self.get_by_name(group_name)
        return common_ldap.filter_entity(group)

    def get_all_filtered(self, hints, query=None):
        if self.ldap_filter:
            query = (query or '') + self.ldap_filter
        query = self.filter_query(hints, query)
        return [common_ldap.filter_entity(group) for group in self.get_all(query, hints)]