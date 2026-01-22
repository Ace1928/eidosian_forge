import types
import testtools
from testtools.content import text_content
from testtools.testcase import skipIf
import fixtures
from fixtures.fixture import gather_details
from fixtures.tests.helpers import LoggingFixture
class TestMethodFixture(testtools.TestCase):

    def test_no_setup_cleanup(self):

        class Stub:
            pass
        fixture = fixtures.MethodFixture(Stub())
        fixture.setUp()
        fixture.reset()
        self.assertIsInstance(fixture.obj, Stub)
        fixture.cleanUp()

    def test_setup_only(self):

        class Stub:

            def setUp(self):
                self.value = 42
        fixture = fixtures.MethodFixture(Stub())
        fixture.setUp()
        self.assertEqual(42, fixture.obj.value)
        self.assertIsInstance(fixture.obj, Stub)
        fixture.cleanUp()

    def test_cleanup_only(self):

        class Stub:
            value = None

            def tearDown(self):
                self.value = 42
        fixture = fixtures.MethodFixture(Stub())
        fixture.setUp()
        self.assertEqual(None, fixture.obj.value)
        self.assertIsInstance(fixture.obj, Stub)
        fixture.cleanUp()
        self.assertEqual(42, fixture.obj.value)

    def test_cleanup(self):

        class Stub:

            def setUp(self):
                self.value = 42

            def tearDown(self):
                self.value = 84
        fixture = fixtures.MethodFixture(Stub())
        fixture.setUp()
        self.assertEqual(42, fixture.obj.value)
        self.assertIsInstance(fixture.obj, Stub)
        fixture.cleanUp()
        self.assertEqual(84, fixture.obj.value)

    def test_custom_setUp(self):

        class Stub:

            def mysetup(self):
                self.value = 42
        obj = Stub()
        fixture = fixtures.MethodFixture(obj, setup=obj.mysetup)
        fixture.setUp()
        self.assertEqual(42, fixture.obj.value)
        self.assertEqual(obj, fixture.obj)
        fixture.cleanUp()

    def test_custom_cleanUp(self):

        class Stub:
            value = 42

            def mycleanup(self):
                self.value = None
        obj = Stub()
        fixture = fixtures.MethodFixture(obj, cleanup=obj.mycleanup)
        fixture.setUp()
        self.assertEqual(42, fixture.obj.value)
        self.assertEqual(obj, fixture.obj)
        fixture.cleanUp()
        self.assertEqual(None, fixture.obj.value)

    def test_reset(self):

        class Stub:

            def setUp(self):
                self.value = 42

            def tearDown(self):
                self.value = 84

            def reset(self):
                self.value = 126
        obj = Stub()
        fixture = fixtures.MethodFixture(obj, reset=obj.reset)
        fixture.setUp()
        self.assertEqual(obj, fixture.obj)
        self.assertEqual(42, obj.value)
        fixture.reset()
        self.assertEqual(126, obj.value)
        fixture.cleanUp()
        self.assertEqual(84, obj.value)