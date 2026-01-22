from libcloud.dns.base import Zone, Record, DNSDriver, RecordType
from libcloud.dns.types import (
from libcloud.utils.py3 import urlencode
from libcloud.common.dnspod import DNSPodResponse, DNSPodException, DNSPodConnection
class DNSPodDNSConnection(DNSPodConnection):
    responseCls = DNSPodDNSResponse