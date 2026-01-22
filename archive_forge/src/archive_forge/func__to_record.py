from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def _to_record(self, data, zone_id=None, zone=None):
    if not zone:
        zone = self.get_zone(zone_id)
    record = data.get('zone_record')
    id = record.get('id')
    name = record.get('name')
    type = record.get('record_type')
    data = record.get('data')
    extra = {'ttl': record.get('ttl'), 'zone_id': record.get('zone_id'), 'aux': record.get('aux')}
    return Record(id=id, name=name, type=type, data=data, zone=zone, driver=self, ttl=record.get('ttl', None), extra=extra)