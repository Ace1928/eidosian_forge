from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
class GoDaddyLegalAgreement:

    def __init__(self, agreement_key, title, url, content):
        self.agreement_key = agreement_key
        self.title = title
        self.url = url
        self.content = content