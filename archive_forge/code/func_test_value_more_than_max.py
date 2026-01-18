import concurrent.futures
import hashlib
import logging
import sys
from unittest import mock
import fixtures
import os_service_types
import testtools
import openstack
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def test_value_more_than_max(self):
    self.assertEqual('1.99', utils.maximum_supported_microversion(self.adapter, '1.100'))