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
def test_obj_as_admin(self):
    self.skipTest('oslo.context does not support elevated()')
    obj = MyObj(context=self.context)

    def fake(*args, **kwargs):
        self.assertTrue(obj._context.is_admin)
    with mock.patch.object(obj, 'obj_reset_changes') as mock_fn:
        mock_fn.side_effect = fake
        with obj.obj_as_admin():
            obj.save()
        self.assertTrue(mock_fn.called)
    self.assertFalse(obj._context.is_admin)