import collections
import copy
import datetime
import hashlib
import inspect
from unittest import mock
import iso8601
from oslo_versionedobjects import base
from oslo_versionedobjects import exception
from oslo_versionedobjects import fields
from oslo_versionedobjects import fixture
from oslo_versionedobjects import test
def test_test_compatibility_routines(self):
    del self.ovc.obj_classes[MyObject2.__name__]
    with mock.patch.object(self.ovc, '_test_object_compatibility') as toc:
        self.ovc.test_compatibility_routines()
    toc.assert_called_once_with(MyObject, manifest=None, init_args=[], init_kwargs={})