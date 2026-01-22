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
def _ldap_get_list(self, search_base, scope, query_params=None, attrlist=None):
    query = u'(objectClass=%s)' % self.object_class
    if query_params:

        def calc_filter(attrname, value):
            val_esc = ldap.filter.escape_filter_chars(value)
            return '(%s=%s)' % (attrname, val_esc)
        query = u'(&%s%s)' % (query, ''.join([calc_filter(k, v) for k, v in query_params.items()]))
    with self.get_connection() as conn:
        return conn.search_s(search_base, scope, query, attrlist)