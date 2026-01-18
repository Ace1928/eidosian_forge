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
def test_obj_repr_unicode(self):
    obj = MyObj(bar='Ƒơơ')
    self.assertEqual("MyObj(bar='Ƒơơ',foo=<?>,missing=<?>,mutable_default=<?>,readonly=<?>,rel_object=<?>,rel_objects=<?>,timestamp=<?>)", repr(obj))