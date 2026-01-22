from __future__ import absolute_import, division, print_function
import re
import traceback
from ansible.module_utils.common.text.converters import to_native
def _find_dn(self):
    dn = self.module.params['dn']
    explode_dn = ldap.dn.explode_dn(dn)
    if len(explode_dn) > 1:
        try:
            escaped_value = ldap.filter.escape_filter_chars(explode_dn[0])
            filterstr = '(%s)' % escaped_value
            dns = self.connection.search_s(','.join(explode_dn[1:]), ldap.SCOPE_ONELEVEL, filterstr)
            if len(dns) == 1:
                dn, dummy = dns[0]
        except Exception:
            pass
    return dn