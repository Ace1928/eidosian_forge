import abc
import codecs
import os.path
import random
import re
import sys
import uuid
import weakref
import ldap.controls
import ldap.filter
import ldappool
from oslo_log import log
from oslo_utils import reflection
from keystone.common import driver_hints
from keystone import exception
from keystone.i18n import _
class EnabledEmuMixIn(BaseLdap):
    """Emulates boolean 'enabled' attribute if turned on.

    Creates a group holding all enabled objects of this class, all missing
    objects are considered disabled.

    Options:

    * $name_enabled_emulation - boolean, on/off
    * $name_enabled_emulation_dn - DN of that group, default is
      cn=enabled_${name}s,${tree_dn}
    * $name_enabled_emulation_use_group_config - boolean, on/off

    Where ${name}s is the plural of self.options_name ('users' or 'tenants'),
    ${tree_dn} is self.tree_dn.
    """
    DEFAULT_GROUP_OBJECTCLASS = 'groupOfNames'
    DEFAULT_MEMBER_ATTRIBUTE = 'member'
    DEFAULT_GROUP_MEMBERS_ARE_IDS = False

    def __init__(self, conf):
        super(EnabledEmuMixIn, self).__init__(conf)
        enabled_emulation = '%s_enabled_emulation' % self.options_name
        self.enabled_emulation = getattr(conf.ldap, enabled_emulation)
        enabled_emulation_dn = '%s_enabled_emulation_dn' % self.options_name
        self.enabled_emulation_dn = getattr(conf.ldap, enabled_emulation_dn)
        use_group_config = '%s_enabled_emulation_use_group_config' % self.options_name
        self.use_group_config = getattr(conf.ldap, use_group_config)
        if not self.use_group_config:
            self.member_attribute = self.DEFAULT_MEMBER_ATTRIBUTE
            self.group_objectclass = self.DEFAULT_GROUP_OBJECTCLASS
            self.group_members_are_ids = self.DEFAULT_GROUP_MEMBERS_ARE_IDS
        else:
            self.member_attribute = conf.ldap.group_member_attribute
            self.group_objectclass = conf.ldap.group_objectclass
            self.group_members_are_ids = conf.ldap.group_members_are_ids
        if not self.enabled_emulation_dn:
            naming_attr_name = 'cn'
            naming_attr_value = 'enabled_%ss' % self.options_name
            sub_vals = (naming_attr_name, naming_attr_value, self.tree_dn)
            self.enabled_emulation_dn = '%s=%s,%s' % sub_vals
            naming_attr = (naming_attr_name, [naming_attr_value])
        else:
            naming_dn = ldap.dn.str2dn(self.enabled_emulation_dn)
            naming_rdn = naming_dn[0][0]
            naming_attr = (naming_rdn[0], naming_rdn[1])
        self.enabled_emulation_naming_attr = naming_attr

    def _id_to_member_attribute_value(self, object_id):
        """Convert id to value expected by member_attribute."""
        if self.group_members_are_ids:
            return object_id
        return self._id_to_dn(object_id)

    def _is_id_enabled(self, object_id, conn):
        member_attr_val = self._id_to_member_attribute_value(object_id)
        return self._is_member_enabled(member_attr_val, conn)

    def _is_member_enabled(self, member_attr_val, conn):
        query = '(%s=%s)' % (self.member_attribute, ldap.filter.escape_filter_chars(member_attr_val))
        try:
            enabled_value = conn.search_s(self.enabled_emulation_dn, ldap.SCOPE_BASE, query, attrlist=DN_ONLY)
        except ldap.NO_SUCH_OBJECT:
            return False
        else:
            return bool(enabled_value)

    def _add_enabled(self, object_id):
        member_attr_val = self._id_to_member_attribute_value(object_id)
        with self.get_connection() as conn:
            if not self._is_member_enabled(member_attr_val, conn):
                modlist = [(ldap.MOD_ADD, self.member_attribute, [member_attr_val])]
                try:
                    conn.modify_s(self.enabled_emulation_dn, modlist)
                except ldap.NO_SUCH_OBJECT:
                    attr_list = [('objectClass', [self.group_objectclass]), (self.member_attribute, [member_attr_val]), self.enabled_emulation_naming_attr]
                    conn.add_s(self.enabled_emulation_dn, attr_list)

    def _remove_enabled(self, object_id):
        member_attr_val = self._id_to_member_attribute_value(object_id)
        modlist = [(ldap.MOD_DELETE, self.member_attribute, [member_attr_val])]
        with self.get_connection() as conn:
            try:
                conn.modify_s(self.enabled_emulation_dn, modlist)
            except (ldap.NO_SUCH_OBJECT, ldap.NO_SUCH_ATTRIBUTE):
                pass

    def create(self, values):
        if self.enabled_emulation:
            enabled_value = values.pop('enabled', True)
            ref = super(EnabledEmuMixIn, self).create(values)
            if 'enabled' not in self.attribute_ignore:
                if enabled_value:
                    self._add_enabled(ref['id'])
                ref['enabled'] = enabled_value
            return ref
        else:
            return super(EnabledEmuMixIn, self).create(values)

    def get(self, object_id, ldap_filter=None):
        with self.get_connection() as conn:
            ref = super(EnabledEmuMixIn, self).get(object_id, ldap_filter)
            if 'enabled' not in self.attribute_ignore and self.enabled_emulation:
                ref['enabled'] = self._is_id_enabled(object_id, conn)
            return ref

    def get_all(self, ldap_filter=None, hints=None):
        hints = hints or driver_hints.Hints()
        if 'enabled' not in self.attribute_ignore and self.enabled_emulation:
            obj_list = [self._ldap_res_to_model(x) for x in self._ldap_get_all(hints, ldap_filter) if x[0] != self.enabled_emulation_dn]
            with self.get_connection() as conn:
                for obj_ref in obj_list:
                    obj_ref['enabled'] = self._is_id_enabled(obj_ref['id'], conn)
            return obj_list
        else:
            return super(EnabledEmuMixIn, self).get_all(ldap_filter, hints)

    def update(self, object_id, values, old_obj=None):
        if 'enabled' not in self.attribute_ignore and self.enabled_emulation:
            data = values.copy()
            enabled_value = data.pop('enabled', None)
            ref = super(EnabledEmuMixIn, self).update(object_id, data, old_obj)
            if enabled_value is not None:
                if enabled_value:
                    self._add_enabled(object_id)
                else:
                    self._remove_enabled(object_id)
                ref['enabled'] = enabled_value
            return ref
        else:
            return super(EnabledEmuMixIn, self).update(object_id, values, old_obj)