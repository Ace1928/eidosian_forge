from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import Provider, RecordType, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError
class GoDaddyDNSResponse(JsonResponse):
    valid_response_codes = [httplib.OK, httplib.ACCEPTED, httplib.CREATED, httplib.NO_CONTENT]

    def parse_body(self):
        if not self.body:
            return None
        self.body = self.body.replace('\\.', '\\\\.')
        data = json.loads(self.body)
        return data

    def parse_error(self):
        data = self.parse_body()
        raise GoDaddyDNSException(code=data['code'], message=data['message'])

    def success(self):
        return self.status in self.valid_response_codes