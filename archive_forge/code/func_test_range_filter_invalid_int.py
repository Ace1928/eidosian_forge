from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_range_filter_invalid_int(self):
    with testtools.ExpectedException(exceptions.SDKException, 'Invalid range value: <1A0'):
        _utils.range_filter(RANGE_DATA, 'key1', '<1A0')