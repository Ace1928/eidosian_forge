from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.common.pointdns import PointDNSConnection
from libcloud.common.exceptions import BaseHTTPError
def _to_mail_redirect(self, data, zone_id=None, zone=None):
    if not zone:
        zone = self.get_zone(zone_id)
    record = data.get('zone_mail_redirect')
    id = record.get('id')
    destination = record.get('destination_address')
    source = record.get('source_address')
    return MailRedirect(id, source, destination, zone, self)