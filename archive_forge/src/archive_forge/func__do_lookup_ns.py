from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
def _do_lookup_ns(self, target):
    nameserver_ips = self.default_nameservers
    nameservers = None
    for i in range(2, len(target.labels) + 1):
        target_part = target.split(i)[1]
        _nameservers = self.cache.get((str(target_part), 'ns'))
        if _nameservers is None:
            nameserver_names, cname = self._lookup_ns_names(target_part, nameservers=nameservers, nameserver_ips=nameserver_ips)
            if nameserver_names is not None:
                nameservers = nameserver_names
            self.cache[str(target_part), 'ns'] = nameservers
            self.cache[str(target_part), 'cname'] = cname
        else:
            nameservers = _nameservers
        nameserver_ips = None
    return nameservers