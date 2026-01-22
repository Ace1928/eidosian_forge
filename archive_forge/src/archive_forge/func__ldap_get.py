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
def _ldap_get(self, object_id, ldap_filter=None):
    query = u'(&(%(id_attr)s=%(id)s)%(filter)s(objectClass=%(object_class)s))' % {'id_attr': self.id_attr, 'id': ldap.filter.escape_filter_chars(str(object_id)), 'filter': ldap_filter or self.ldap_filter or '', 'object_class': self.object_class}
    with self.get_connection() as conn:
        try:
            attrs = list(set([self.id_attr] + list(self.attribute_mapping.values()) + list(self.extra_attr_mapping.keys())))
            res = conn.search_s(self.tree_dn, self.LDAP_SCOPE, query, attrs)
        except ldap.NO_SUCH_OBJECT:
            return None
    try:
        return self._filter_ldap_result_by_attr(res[:1], 'name')[0]
    except IndexError:
        return None