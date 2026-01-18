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
def test_obj_tracking(self):

    @base.VersionedObjectRegistry.register
    class NewBaseClass(object):
        VERSION = '1.0'
        fields = {}

        @classmethod
        def obj_name(cls):
            return cls.__name__

    @base.VersionedObjectRegistry.register
    class Fake1TestObj1(NewBaseClass):

        @classmethod
        def obj_name(cls):
            return 'fake1'

    @base.VersionedObjectRegistry.register
    class Fake1TestObj2(Fake1TestObj1):
        pass

    @base.VersionedObjectRegistry.register
    class Fake1TestObj3(Fake1TestObj1):
        VERSION = '1.1'

    @base.VersionedObjectRegistry.register
    class Fake2TestObj1(NewBaseClass):

        @classmethod
        def obj_name(cls):
            return 'fake2'

    @base.VersionedObjectRegistry.register
    class Fake1TestObj4(Fake1TestObj3):
        VERSION = '1.2'

    @base.VersionedObjectRegistry.register
    class Fake2TestObj2(Fake2TestObj1):
        VERSION = '1.1'

    @base.VersionedObjectRegistry.register
    class Fake1TestObj5(Fake1TestObj1):
        VERSION = '1.1'

    @base.VersionedObjectRegistry.register_if(False)
    class ConditionalObj1(NewBaseClass):
        fields = {'foo': fields.IntegerField()}

    @base.VersionedObjectRegistry.register_if(True)
    class ConditionalObj2(NewBaseClass):
        fields = {'foo': fields.IntegerField()}
    expected = {'fake1': [Fake1TestObj4, Fake1TestObj5, Fake1TestObj2], 'fake2': [Fake2TestObj2, Fake2TestObj1]}
    self.assertEqual(expected['fake1'], base.VersionedObjectRegistry.obj_classes()['fake1'])
    self.assertEqual(expected['fake2'], base.VersionedObjectRegistry.obj_classes()['fake2'])
    self.assertEqual([], base.VersionedObjectRegistry.obj_classes()['ConditionalObj1'])
    self.assertTrue(hasattr(ConditionalObj1, 'foo'))
    self.assertEqual([ConditionalObj2], base.VersionedObjectRegistry.obj_classes()['ConditionalObj2'])
    self.assertTrue(hasattr(ConditionalObj2, 'foo'))