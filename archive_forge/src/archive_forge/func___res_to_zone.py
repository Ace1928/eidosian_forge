import hmac
import json
import base64
import datetime
from hashlib import sha256
from libcloud.dns.base import Zone, Record, DNSDriver
from libcloud.dns.types import (
from libcloud.utils.py3 import b, httplib
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import LibcloudError, ProviderError, InvalidCredsError
def __res_to_zone(self, zone):
    return Zone(id=zone['id'], domain=zone['name'], type=DEFAULT_ZONE_TYPE, ttl=DEFAULT_ZONE_TTL, driver=self.connection.driver, extra={'created': zone['created'], 'servers': zone['servers'], 'account_id': zone['account_id'], 'cluster_id': zone['cluster_id']})