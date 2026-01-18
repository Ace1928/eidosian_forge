import re
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.common.types import LibcloudError
from libcloud.common.worldwidedns import WorldWideDNSConnection
Return an available entry to store a record.