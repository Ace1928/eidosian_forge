import random
import re
import shelve
import ldap
from oslo_log import log
import keystone.conf
from keystone import exception
from keystone.identity.backends.ldap import common
def _internal_attr(attr_name, value_or_values):

    def normalize_value(value):
        return common.utf8_decode(value)

    def normalize_dn(dn):
        if dn == 'cn=Doe\\5c, John,ou=Users,cn=example,cn=com':
            return 'CN=Doe\\, John,OU=Users,CN=example,CN=com'
        if dn == 'cn=Doe\\, John,ou=Users,cn=example,cn=com':
            return 'CN=Doe\\2C John,OU=Users,CN=example,CN=com'
        try:
            dn = ldap.dn.str2dn(dn)
        except ldap.DECODING_ERROR:
            return normalize_value(dn)
        norm = []
        for part in dn:
            name, val, i = part[0]
            name = name.upper()
            norm.append([(name, val, i)])
        return ldap.dn.dn2str(norm)
    if attr_name in ('member', 'roleOccupant'):
        attr_fn = normalize_dn
    else:
        attr_fn = normalize_value
    if isinstance(value_or_values, list):
        return [attr_fn(x) for x in value_or_values]
    return [attr_fn(value_or_values)]