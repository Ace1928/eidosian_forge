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
def test_subclassability(self):

    class MyRegistryOne(base.VersionedObjectRegistry):

        def registration_hook(self, cls, index):
            cls.reg_to = 'one'

    class MyRegistryTwo(base.VersionedObjectRegistry):

        def registration_hook(self, cls, index):
            cls.reg_to = 'two'

    @MyRegistryOne.register
    class AVersionedObject1(base.VersionedObject):
        VERSION = '1.0'
        fields = {'baz': fields.Field(fields.Integer())}

    @MyRegistryTwo.register
    class AVersionedObject2(base.VersionedObject):
        VERSION = '1.0'
        fields = {'baz': fields.Field(fields.Integer())}
    self.assertIn('AVersionedObject1', MyRegistryOne.obj_classes())
    self.assertIn('AVersionedObject2', MyRegistryOne.obj_classes())
    self.assertIn('AVersionedObject1', MyRegistryTwo.obj_classes())
    self.assertIn('AVersionedObject2', MyRegistryTwo.obj_classes())
    self.assertIn('AVersionedObject1', base.VersionedObjectRegistry.obj_classes())
    self.assertIn('AVersionedObject2', base.VersionedObjectRegistry.obj_classes())
    self.assertEqual(AVersionedObject1.reg_to, 'one')
    self.assertEqual(AVersionedObject2.reg_to, 'two')