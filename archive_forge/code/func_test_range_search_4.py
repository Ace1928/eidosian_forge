from unittest import mock
import uuid
import testtools
from openstack import connection
from openstack import exceptions
from openstack.tests.unit import base
from openstack import utils
def test_range_search_4(self):
    filters = {'key1': 'max', 'key2': 'min'}
    retval = self.cloud.range_search(RANGE_DATA, filters)
    self.assertIsInstance(retval, list)
    self.assertEqual(0, len(retval))