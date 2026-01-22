import sys
import unittest
from libcloud.test import MockHttp
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, RecordDoesNotExistError
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.test.secrets import DNS_PARAMS_ZERIGO
from libcloud.dns.drivers.zerigo import ZerigoError, ZerigoDNSDriver
from libcloud.test.file_fixtures import DNSFileFixtures
def _api_1_1_zones_xml_INVALID_CREDS(self, method, url, body, headers):
    body = 'HTTP Basic: Access denied.\n'
    return (httplib.UNAUTHORIZED, body, {}, httplib.responses[httplib.OK])