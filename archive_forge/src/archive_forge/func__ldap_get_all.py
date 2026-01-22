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
@driver_hints.truncated
def _ldap_get_all(self, hints, ldap_filter=None):
    query = u'(&%s(objectClass=%s)(%s=*))' % (ldap_filter or self.ldap_filter or '', self.object_class, self.id_attr)
    sizelimit = 0
    attrs = list(set([self.id_attr] + list(self.attribute_mapping.values()) + list(self.extra_attr_mapping.keys())))
    if hints.limit:
        sizelimit = hints.limit['limit']
        res = self._ldap_get_limited(self.tree_dn, self.LDAP_SCOPE, query, attrs, sizelimit)
    else:
        with self.get_connection() as conn:
            try:
                res = conn.search_s(self.tree_dn, self.LDAP_SCOPE, query, attrs)
            except ldap.NO_SUCH_OBJECT:
                return []
    return self._filter_ldap_result_by_attr(res, 'name')