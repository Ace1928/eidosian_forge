from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.six import raise_from
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.dns.plugins.module_utils.record import (
from ansible_collections.community.dns.plugins.module_utils.wsdl import (
from ansible_collections.community.dns.plugins.module_utils.zone import (
from ansible_collections.community.dns.plugins.module_utils.zone_record_api import (
def _create_record_from_encoding(source, type=None):
    source = dict(source)
    result = DNSRecord()
    result.id = source.pop('id')
    result.type = source.pop('type', type)
    result.prefix = source.pop('prefix', None)
    ttl = source.pop('ttl')
    result.ttl = int(ttl) if ttl is not None else None
    priority = source.pop('priority')
    target = source.pop('target')
    if result.type in ('PTR', 'MX'):
        result.target = '{0} {1}'.format(priority, target)
    else:
        result.target = target
    source.pop('zone', None)
    result.extra['comment'] = source.pop('comment') or ''
    result.extra.update(source)
    return result