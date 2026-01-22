import re
import sys
import datetime
import unittest
import traceback
from unittest.mock import patch, mock_open
from libcloud.test import MockHttp
from libcloud.utils.py3 import ET, PY2, b, httplib, assertRaisesRegex
from libcloud.compute.base import Node, NodeImage
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import VCLOUD_PARAMS
from libcloud.compute.types import NodeState
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vcloud import (
def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_action_controlAccess(self, method, url, body, headers):
    body = str(body)
    self.assertTrue(method == 'POST')
    self.assertTrue('<IsSharedToEveryone>false</IsSharedToEveryone>' in body)
    self.assertTrue('<Subject href="https://vm-vcloud/api/admin/group/b8202c48-7151-4e61-9a6c-155474c7d413" />' in body)
    self.assertTrue('<AccessLevel>FullControl</AccessLevel>' in body)
    body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6a_controlAccess.xml')
    return (httplib.OK, body, headers, httplib.responses[httplib.OK])