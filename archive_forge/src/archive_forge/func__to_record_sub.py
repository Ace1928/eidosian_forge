import copy
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.gandi_live import (
def _to_record_sub(self, data, zone, value):
    extra = {}
    ttl = data.get('rrset_ttl', None)
    if ttl is not None:
        extra['ttl'] = int(ttl)
    if data['rrset_type'] == 'MX':
        priority, value = value.split()
        extra['priority'] = priority
    return Record(id='{}:{}'.format(data['rrset_type'], data['rrset_name']), name=data['rrset_name'], type=self._string_to_record_type(data['rrset_type']), data=value, zone=zone, driver=self, ttl=ttl, extra=extra)