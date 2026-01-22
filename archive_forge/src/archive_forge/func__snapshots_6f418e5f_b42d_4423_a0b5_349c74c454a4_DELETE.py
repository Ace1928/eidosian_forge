import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import SCALEWAY_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.scaleway import ScalewayNodeDriver
def _snapshots_6f418e5f_b42d_4423_a0b5_349c74c454a4_DELETE(self, method, url, body, headers):
    return (httplib.NO_CONTENT, None, {}, httplib.responses[httplib.NO_CONTENT])