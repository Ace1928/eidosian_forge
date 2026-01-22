from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
class GoDaddyDomainPurchaseResponse:

    def __init__(self, order_id, item_count, total, currency):
        self.order_id = order_id
        self.item_count = item_count
        self.total = total
        self.current = currency