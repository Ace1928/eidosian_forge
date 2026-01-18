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
Fixtures needed for tests related to rating model