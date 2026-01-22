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
def _json_1_0_cloud_project_project_id_instance_get_invalid_app_key_error(self, method, url, body, headers):
    body = '{"message":"Invalid application key"}'
    return (httplib.UNAUTHORIZED, body, {}, httplib.responses[httplib.OK])