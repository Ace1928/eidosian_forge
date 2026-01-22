from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_native, to_text
def _get_resolver(self, dnsname, nameservers):
    cache_index = ('|'.join([str(dnsname)] + sorted(nameservers)), 'resolver')
    resolver = self.cache.get(cache_index)
    if resolver is None:
        resolver = dns.resolver.Resolver(configure=False)
        resolver.use_edns(0, ednsflags=dns.flags.DO, payload=_EDNS_SIZE)
        resolver.timeout = self.timeout
        nameserver_ips = set()
        for nameserver in nameservers:
            nameserver_ips.update(self._lookup_address(nameserver))
        resolver.nameservers = sorted(nameserver_ips)
        self.cache[cache_index] = resolver
    return resolver