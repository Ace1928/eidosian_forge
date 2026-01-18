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
@mock.patch.object(base, '_make_class_properties')
def test_register_if_false(self, mock_make_props):

    class my_class(object):
        pass
    base.VersionedObjectRegistry.register_if(False)(my_class)
    mock_make_props.assert_called_once_with(my_class)