import sys
import json
import functools
from datetime import datetime
from unittest import mock
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import httplib, parse_qs, urlparse, urlunquote
from libcloud.common.types import LibcloudError
from libcloud.compute.base import NodeSize, NodeLocation, StorageVolume, VolumeSnapshot
from libcloud.compute.types import Provider, NodeState, StorageVolumeState, VolumeSnapshotState
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.azure_arm import (
def test_update_nic_properties(self):
    nics = self.driver.ex_list_nics()
    nic_to_update = nics[0]
    nic_properties = nic_to_update.extra
    ip_configs = nic_properties['ipConfigurations']
    ip_configs[0]['properties']['primary'] = True
    updated_nic = self.driver.ex_update_nic_properties(nic_to_update, resource_group='REVIZOR', properties=nic_properties)
    self.assertTrue(updated_nic.extra['ipConfigurations'][0]['properties']['primary'])