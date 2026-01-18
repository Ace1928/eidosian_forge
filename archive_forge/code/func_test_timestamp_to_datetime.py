import sys
import datetime
import unittest
from unittest import mock
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import GCE_PARAMS, GCE_KEYWORD_PARAMS
from libcloud.common.google import (
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gce import (
from libcloud.test.common.test_google import GoogleTestCase, GoogleAuthMockHttp
def test_timestamp_to_datetime(self):
    timestamp1 = '2013-06-26T10:05:19.340-07:00'
    datetime1 = datetime.datetime(2013, 6, 26, 17, 5, 19)
    self.assertEqual(timestamp_to_datetime(timestamp1), datetime1)
    timestamp2 = '2013-06-26T17:43:15.000-00:00'
    datetime2 = datetime.datetime(2013, 6, 26, 17, 43, 15)
    self.assertEqual(timestamp_to_datetime(timestamp2), datetime2)