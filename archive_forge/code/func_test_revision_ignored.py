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
@mock.patch('oslo_versionedobjects.base.obj_tree_get_versions')
def test_revision_ignored(self, mock_otgv):
    mock_otgv.return_value = {'MyObj': '1.1.456'}
    obj = MyObj2.query(self.context)
    self.assertEqual('bar', obj.bar)