import os
import sys
import datetime
import unittest
from unittest import mock
from unittest.mock import Mock, patch
import pytest
import requests_mock
from libcloud.test import XML_HEADERS, MockHttp
from libcloud.pricing import set_pricing, clear_pricing_data
from libcloud.utils.py3 import u, httplib, method_type
from libcloud.common.base import LibcloudConnection
from libcloud.common.types import LibcloudError, InvalidCredsError, MalformedResponseError
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import OPENSTACK_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import OpenStackFixtures, ComputeFileFixtures
from libcloud.common.openstack_identity import (
from libcloud.compute.drivers.openstack import (
def _v2_1337_v2_0_floatingips_09ea1784_2f81_46dc_8c91_244b4df75bde(self, method, url, body, headers):
    if method == 'PUT':
        self.assertIn(body, ['{"floatingip": {"port_id": "ce531f90-199f-48c0-816c-13e38010b442"}}', '{"floatingip": {"port_id": null}}'])
        body = ''
        return (httplib.OK, body, self.json_content_headers, httplib.responses[httplib.OK])