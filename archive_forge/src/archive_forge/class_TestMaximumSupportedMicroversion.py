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
class TestMaximumSupportedMicroversion(base.TestCase):

    def setUp(self):
        super(TestMaximumSupportedMicroversion, self).setUp()
        self.adapter = mock.Mock(spec=['get_endpoint_data'])
        self.endpoint_data = mock.Mock(spec=['min_microversion', 'max_microversion'], min_microversion=None, max_microversion='1.99')
        self.adapter.get_endpoint_data.return_value = self.endpoint_data

    def test_with_none(self):
        self.assertIsNone(utils.maximum_supported_microversion(self.adapter, None))

    def test_with_value(self):
        self.assertEqual('1.42', utils.maximum_supported_microversion(self.adapter, '1.42'))

    def test_value_more_than_max(self):
        self.assertEqual('1.99', utils.maximum_supported_microversion(self.adapter, '1.100'))

    def test_value_less_than_min(self):
        self.endpoint_data.min_microversion = '1.42'
        self.assertIsNone(utils.maximum_supported_microversion(self.adapter, '1.2'))