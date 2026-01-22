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
def _dn_to_id(self, dn):
    if self.id_attr == ldap.dn.str2dn(dn)[0][0][0].lower():
        return ldap.dn.str2dn(dn)[0][0][1]
    else:
        with self.get_connection() as conn:
            search_result = conn.search_s(dn, ldap.SCOPE_BASE)
        if search_result:
            try:
                id_list = search_result[0][1][self.id_attr]
            except KeyError:
                message = 'ID attribute %(id_attr)s not found in LDAP object %(dn)s.' % {'id_attr': self.id_attr, 'dn': search_result}
                LOG.warning(message)
                raise exception.NotFound(message=message)
            if len(id_list) > 1:
                message = 'In order to keep backward compatibility, in the case of multivalued ids, we are returning the first id %(id_attr)s in the DN.' % {'id_attr': id_list[0]}
                LOG.warning(message)
            return id_list[0]
        else:
            message = _('DN attribute %(dn)s not found in LDAP') % {'dn': dn}
            raise exception.NotFound(message=message)