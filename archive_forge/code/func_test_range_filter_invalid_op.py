from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_range_filter_invalid_op(self):
    with testtools.ExpectedException(exceptions.SDKException, 'Invalid range value: <>100'):
        _utils.range_filter(RANGE_DATA, 'key1', '<>100')