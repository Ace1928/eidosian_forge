from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def _to_zone(self, data):
    zone = data.get('zone')
    id = zone.get('id')
    name = zone.get('name')
    ttl = zone.get('ttl')
    extra = {'group': zone.get('group'), 'user-id': zone.get('user-id')}
    type = 'master'
    return Zone(id=id, domain=name, type=type, ttl=ttl, driver=self, extra=extra)