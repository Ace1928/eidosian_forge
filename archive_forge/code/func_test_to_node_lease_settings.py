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
def test_to_node_lease_settings(self):
    node = self.driver._ex_get_node('https://vm-vcloud/api/vApp/vapp-8c57a5b6-e61b-48ca-8a78-3b70ee65ef6a')
    lease = Lease('https://vm-vcloud/api/vApp/vapp-8c57a5b6-e61b-48ca-8a78-3b70ee65ef6a/leaseSettingsSection/', deployment_lease=0, storage_lease=0)
    self.assertEqual(node.extra['lease_settings'], lease)
    node = self.driver._ex_get_node('https://vm-vcloud/api/vApp/vapp-8c57a5b6-e61b-48ca-8a78-3b70ee65ef6d')
    lease = Lease('https://vm-vcloud/api/vApp/vapp-8c57a5b6-e61b-48ca-8a78-3b70ee65ef6d/leaseSettingsSection/', deployment_lease=86400, storage_lease=172800, deployment_lease_expiration=datetime.datetime(year=2019, month=10, day=7, hour=14, minute=6, second=29, microsecond=980725, tzinfo=UTC), storage_lease_expiration=datetime.datetime(year=2019, month=10, day=8, hour=14, minute=6, second=29, microsecond=980725, tzinfo=UTC))
    self.assertEqual(node.extra['lease_settings'], lease)