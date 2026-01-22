import sys
import json
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.dns.base import Zone, Record
from libcloud.dns.types import RecordType, ZoneDoesNotExistError, ZoneAlreadyExistsError
from libcloud.utils.py3 import httplib
from libcloud.test.file_fixtures import DNSFileFixtures
from libcloud.dns.drivers.rcodezero import RcodeZeroDNSDriver
def _api_v1_zones_example_com_MISSING(self, *args, **kwargs):
    return (httplib.NOT_FOUND, '{"status": "failed","message": "Zone not found"}', self.base_headers, 'Unprocessable Entity')