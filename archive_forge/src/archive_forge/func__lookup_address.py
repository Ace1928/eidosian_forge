from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
def _lookup_address(self, target):
    result = self.cache.get((target, 'addr'))
    if not result:
        result = self._lookup_address_impl(target, dns.rdatatype.A)
        result.extend(self._lookup_address_impl(target, dns.rdatatype.AAAA))
        self.cache[target, 'addr'] = result
    return result