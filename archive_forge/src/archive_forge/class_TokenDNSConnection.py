import json
import itertools
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.misc import reverse_dict, merge_valid_keys
from libcloud.common.base import JsonResponse, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, InvalidCredsError
class TokenDNSConnection(BaseDNSConnection, ConnectionKey):

    def add_default_headers(self, headers):
        headers['Content-Type'] = 'application/json'
        headers['Authorization'] = 'Bearer %s' % self.key
        return headers