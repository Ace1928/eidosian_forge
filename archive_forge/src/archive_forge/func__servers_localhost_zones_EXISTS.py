import sys
import json
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.powerdns import PowerDNSDriver
def _servers_localhost_zones_EXISTS(self, method, url, body, headers):
    if method != 'POST':
        raise NotImplementedError('Unexpected method: %s' % method)
    payload = json.loads(body)
    domain = payload['name']
    body = json.dumps({'error': "Domain '%s' already exists" % domain})
    return (httplib.UNPROCESSABLE_ENTITY, body, self.base_headers, 'Unprocessable Entity')