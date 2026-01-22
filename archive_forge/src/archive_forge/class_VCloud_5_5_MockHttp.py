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
class VCloud_5_5_MockHttp(VCloud_1_5_MockHttp):

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_action_createSnapshot(self, method, url, body, headers):
        assert method == 'POST'
        body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_create_snapshot.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_task_fab4b26f_4f2e_4d49_ad01_ae9324bbfe48(self, method, url, body, headers):
        body = self.fixtures.load('api_task_b034df55_fe81_4798_bc81_1f0fd0ead450.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_action_removeAllSnapshots(self, method, url, body, headers):
        assert method == 'POST'
        body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_remove_snapshots.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_task_2518935e_b315_4d8e_9e99_9275f751877c(self, method, url, body, headers):
        body = self.fixtures.load('api_task_2518935e_b315_4d8e_9e99_9275f751877c.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_action_revertToCurrentSnapshot(self, method, url, body, headers):
        assert method == 'POST'
        body = self.fixtures.load('api_vApp_vapp_8c57a5b6_e61b_48ca_8a78_3b70ee65ef6b_revert_snapshot.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])

    def _api_task_fe75d3af_f5a3_44a5_b016_ae0bdadfc32b(self, method, url, body, headers):
        body = self.fixtures.load('api_task_fe75d3af_f5a3_44a5_b016_ae0bdadfc32b.xml')
        return (httplib.OK, body, headers, httplib.responses[httplib.OK])