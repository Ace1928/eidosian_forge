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
def _console_tests(self, level, debug, stream):
    openstack.enable_logging(debug=debug, stream=stream)
    self.assertEqual(self.openstack_logger.addHandler.call_count, 1)
    self.openstack_logger.setLevel.assert_called_with(level)