from unittest import mock
from uuid import uuid4
import testtools
from openstack.cloud import _utils
from openstack import exceptions
from openstack.tests.unit import base
def test_safe_dict_max_not_int(self):
    """Test non-integer key value raises OSCE"""
    data = [{'f1': 3}, {'f1': 'aaa'}, {'f1': 1}]
    with testtools.ExpectedException(exceptions.SDKException, 'Search for maximum value failed. Value for f1 is not an integer: aaa'):
        _utils.safe_dict_max('f1', data)