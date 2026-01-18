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
def test_requested_not_supported_no_default_exception(self):
    self.adapter.default_microversion = None
    self.assertRaises(exceptions.SDKException, utils.supports_microversion, self.adapter, '2.2', True)