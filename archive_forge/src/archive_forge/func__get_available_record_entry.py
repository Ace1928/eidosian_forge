import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.types import LibcloudError
from libcloud.common.worldwidedns import WorldWideDNSConnection
def _get_available_record_entry(self, zone):
    """Return an available entry to store a record."""
    entries = zone.extra
    for entry in range(1, MAX_RECORD_ENTRIES + 1):
        subdomain = entries.get('S%s' % entry)
        _type = entries.get('T%s' % entry)
        data = entries.get('D%s' % entry)
        if not any([subdomain, _type, data]):
            return entry
    return None