from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
class GoDaddyAvailability:

    def __init__(self, domain, available, price, currency, period):
        self.domain = domain
        self.available = bool(available)
        self.price = float(price) / 1000000
        self.currency = currency
        self.period = int(period)