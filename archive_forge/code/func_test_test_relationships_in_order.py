import copy
import datetime
import jsonschema
import logging
import pytz
from unittest import mock
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import timeutils
import testtools
from testtools import matchers
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_test_relationships_in_order(self):
    fake_classes = {mock.sentinel.class_one: [mock.sentinel.impl_one_one, mock.sentinel.impl_one_two], mock.sentinel.class_two: [mock.sentinel.impl_two_one, mock.sentinel.impl_two_two]}
    checker = fixture.ObjectVersionChecker(fake_classes)

    @mock.patch.object(checker, '_test_relationships_in_order')
    def test(mock_compat):
        checker.test_relationships_in_order()
        mock_compat.assert_has_calls([mock.call(mock.sentinel.impl_one_one), mock.call(mock.sentinel.impl_one_two), mock.call(mock.sentinel.impl_two_one), mock.call(mock.sentinel.impl_two_two)], any_order=True)
    test()