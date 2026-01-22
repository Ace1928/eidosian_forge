from fixtures.tests.helpers import LoggingFixture
import testtools
import testresources
from testresources.tests import (
class TestGenericResource(testtools.TestCase):

    def test_default_uses_setUp_tearDown(self):
        calls = []

        class Wrapped:

            def setUp(self):
                calls.append('setUp')

            def tearDown(self):
                calls.append('tearDown')
        mgr = testresources.GenericResource(Wrapped)
        resource = mgr.getResource()
        self.assertEqual(['setUp'], calls)
        mgr.finishedWith(resource)
        self.assertEqual(['setUp', 'tearDown'], calls)
        self.assertIsInstance(resource, Wrapped)

    def test_dependencies_passed_to_factory(self):
        calls = []

        class Wrapped:

            def __init__(self, **args):
                calls.append(args)

            def setUp(self):
                pass

            def tearDown(self):
                pass

        class Trivial(testresources.TestResource):

            def __init__(self, thing):
                testresources.TestResource.__init__(self)
                self.thing = thing

            def make(self, dependency_resources):
                return self.thing

            def clean(self, resource):
                pass
        mgr = testresources.GenericResource(Wrapped)
        mgr.resources = [('foo', Trivial('foo')), ('bar', Trivial('bar'))]
        resource = mgr.getResource()
        self.assertEqual([{'foo': 'foo', 'bar': 'bar'}], calls)
        mgr.finishedWith(resource)

    def test_setup_teardown_controllable(self):
        calls = []

        class Wrapped:

            def start(self):
                calls.append('setUp')

            def stop(self):
                calls.append('tearDown')
        mgr = testresources.GenericResource(Wrapped, setup_method_name='start', teardown_method_name='stop')
        resource = mgr.getResource()
        self.assertEqual(['setUp'], calls)
        mgr.finishedWith(resource)
        self.assertEqual(['setUp', 'tearDown'], calls)
        self.assertIsInstance(resource, Wrapped)

    def test_always_dirty(self):

        class Wrapped:

            def setUp(self):
                pass

            def tearDown(self):
                pass
        mgr = testresources.GenericResource(Wrapped)
        resource = mgr.getResource()
        self.assertTrue(mgr.isDirty())
        mgr.finishedWith(resource)