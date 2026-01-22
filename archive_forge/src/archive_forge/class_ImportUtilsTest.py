import datetime
import sys
from oslotest import base as test_base
from oslo_utils import importutils
class ImportUtilsTest(test_base.BaseTestCase):

    def test_import_class(self):
        dt = importutils.import_class('datetime.datetime')
        self.assertEqual(sys.modules['datetime'].datetime, dt)

    def test_import_bad_class(self):
        self.assertRaises(ImportError, importutils.import_class, 'lol.u_mad.brah')

    def test_import_module(self):
        dt = importutils.import_module('datetime')
        self.assertEqual(sys.modules['datetime'], dt)

    def test_import_object_optional_arg_not_present(self):
        obj = importutils.import_object('oslo_utils.tests.fake.FakeDriver')
        self.assertEqual(obj.__class__.__name__, 'FakeDriver')

    def test_import_object_optional_arg_present(self):
        obj = importutils.import_object('oslo_utils.tests.fake.FakeDriver', first_arg=False)
        self.assertEqual(obj.__class__.__name__, 'FakeDriver')

    def test_import_object_required_arg_not_present(self):
        self.assertRaises(TypeError, importutils.import_object, 'oslo_utils.tests.fake.FakeDriver2')

    def test_import_object_required_arg_present(self):
        obj = importutils.import_object('oslo_utils.tests.fake.FakeDriver2', first_arg=False)
        self.assertEqual(obj.__class__.__name__, 'FakeDriver2')

    def test_import_object_ns_optional_arg_not_present(self):
        obj = importutils.import_object_ns('oslo_utils', 'tests.fake.FakeDriver')
        self.assertEqual(obj.__class__.__name__, 'FakeDriver')

    def test_import_object_ns_optional_arg_present(self):
        obj = importutils.import_object_ns('oslo_utils', 'tests.fake.FakeDriver', first_arg=False)
        self.assertEqual(obj.__class__.__name__, 'FakeDriver')

    def test_import_object_ns_required_arg_not_present(self):
        self.assertRaises(TypeError, importutils.import_object_ns, 'oslo_utils', 'tests.fake.FakeDriver2')

    def test_import_object_ns_required_arg_present(self):
        obj = importutils.import_object_ns('oslo_utils', 'tests.fake.FakeDriver2', first_arg=False)
        self.assertEqual(obj.__class__.__name__, 'FakeDriver2')

    def test_import_object_ns_full_optional_arg_not_present(self):
        obj = importutils.import_object_ns('tests2', 'oslo_utils.tests.fake.FakeDriver')
        self.assertEqual(obj.__class__.__name__, 'FakeDriver')

    def test_import_object_ns_full_optional_arg_present(self):
        obj = importutils.import_object_ns('tests2', 'oslo_utils.tests.fake.FakeDriver', first_arg=False)
        self.assertEqual(obj.__class__.__name__, 'FakeDriver')

    def test_import_object_ns_full_required_arg_not_present(self):
        self.assertRaises(TypeError, importutils.import_object_ns, 'tests2', 'oslo_utils.tests.fake.FakeDriver2')

    def test_import_object_ns_full_required_arg_present(self):
        obj = importutils.import_object_ns('tests2', 'oslo_utils.tests.fake.FakeDriver2', first_arg=False)
        self.assertEqual(obj.__class__.__name__, 'FakeDriver2')

    def test_import_object_ns_raise_import_error_in_init(self):
        self.assertRaises(ImportError, importutils.import_object_ns, 'tests2', 'oslo_utils.tests.fake.FakeDriver3')

    def test_import_object(self):
        dt = importutils.import_object('datetime.time')
        self.assertIsInstance(dt, sys.modules['datetime'].time)

    def test_import_object_with_args(self):
        dt = importutils.import_object('datetime.datetime', 2012, 4, 5)
        self.assertIsInstance(dt, sys.modules['datetime'].datetime)
        self.assertEqual(dt, datetime.datetime(2012, 4, 5))

    def test_import_versioned_module(self):
        v2 = importutils.import_versioned_module('oslo_utils.tests.fake', 2)
        self.assertEqual(sys.modules['oslo_utils.tests.fake.v2'], v2)
        dummpy = importutils.import_versioned_module('oslo_utils.tests.fake', 2, 'dummpy')
        self.assertEqual(sys.modules['oslo_utils.tests.fake.v2.dummpy'], dummpy)

    def test_import_versioned_module_wrong_version_parameter(self):
        self.assertRaises(ValueError, importutils.import_versioned_module, 'oslo_utils.tests.fake', '2.0', 'fake')

    def test_import_versioned_module_error(self):
        self.assertRaises(ImportError, importutils.import_versioned_module, 'oslo_utils.tests.fake', 2, 'fake')

    def test_try_import(self):
        dt = importutils.try_import('datetime')
        self.assertEqual(sys.modules['datetime'], dt)

    def test_try_import_returns_default(self):
        foo = importutils.try_import('foo.bar')
        self.assertIsNone(foo)

    def test_import_any_none_found(self):
        self.assertRaises(ImportError, importutils.import_any, 'foo.bar', 'foo.foo.bar')

    def test_import_any_found(self):
        dt = importutils.import_any('foo.bar', 'datetime')
        self.assertEqual(sys.modules['datetime'], dt)