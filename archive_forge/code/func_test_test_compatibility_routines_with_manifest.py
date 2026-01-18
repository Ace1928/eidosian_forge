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
def test_test_compatibility_routines_with_manifest(self):
    del self.ovc.obj_classes[MyObject2.__name__]
    man = {'who': 'cares'}
    with mock.patch.object(self.ovc, '_test_object_compatibility') as toc:
        with mock.patch('oslo_versionedobjects.base.obj_tree_get_versions') as otgv:
            otgv.return_value = man
            self.ovc.test_compatibility_routines(use_manifest=True)
    otgv.assert_called_once_with(MyObject.__name__)
    toc.assert_called_once_with(MyObject, manifest=man, init_args=[], init_kwargs={})