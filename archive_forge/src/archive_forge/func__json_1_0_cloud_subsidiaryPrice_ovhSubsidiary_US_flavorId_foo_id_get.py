import sys
import unittest
from unittest.mock import patch
from libcloud.http import LibcloudConnection
from libcloud.test import no_internet
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import OVH_PARAMS
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ovh import OvhNodeDriver
from libcloud.test.common.test_ovh import BaseOvhMockHttp
def _json_1_0_cloud_subsidiaryPrice_ovhSubsidiary_US_flavorId_foo_id_get(self, method, url, body, headers):
    return self._json_1_0_cloud_subsidiaryPrice_flavorId_foo_id_ovhSubsidiary_US_get(method, url, body, headers)