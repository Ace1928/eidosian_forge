from __future__ import annotations
import os
import sys
import types
from typing_extensions import NoReturn
from twisted.python import rebuild
from twisted.trial.unittest import TestCase
from . import crash_test_dummy
class RebuildTests(TestCase):
    """
    Simple testcase for rebuilding, to at least exercise the code.
    """

    def setUp(self) -> None:
        self.libPath = self.mktemp()
        os.mkdir(self.libPath)
        self.fakelibPath = os.path.join(self.libPath, 'twisted_rebuild_fakelib')
        os.mkdir(self.fakelibPath)
        open(os.path.join(self.fakelibPath, '__init__.py'), 'w').close()
        sys.path.insert(0, self.libPath)

    def tearDown(self) -> None:
        sys.path.remove(self.libPath)

    def test_FileRebuild(self) -> None:
        import shutil
        import time
        from twisted.python.util import sibpath
        shutil.copyfile(sibpath(__file__, 'myrebuilder1.py'), os.path.join(self.fakelibPath, 'myrebuilder.py'))
        from twisted_rebuild_fakelib import myrebuilder
        a = myrebuilder.A()
        b = myrebuilder.B()
        i = myrebuilder.Inherit()
        self.assertEqual(a.a(), 'a')
        time.sleep(1.1)
        shutil.copyfile(sibpath(__file__, 'myrebuilder2.py'), os.path.join(self.fakelibPath, 'myrebuilder.py'))
        rebuild.rebuild(myrebuilder)
        b2 = myrebuilder.B()
        self.assertEqual(b2.b(), 'c')
        self.assertEqual(b.b(), 'c')
        self.assertEqual(i.a(), 'd')
        self.assertEqual(a.a(), 'b')

    def test_Rebuild(self) -> None:
        """
        Rebuilding an unchanged module.
        """
        x = crash_test_dummy.X('a')
        rebuild.rebuild(crash_test_dummy, doLog=False)
        x.do()
        self.assertEqual(x.__class__, crash_test_dummy.X)
        self.assertEqual(f, crash_test_dummy.foo)

    def test_ComponentInteraction(self) -> None:
        x = crash_test_dummy.XComponent()
        x.setAdapter(crash_test_dummy.IX, crash_test_dummy.XA)
        x.getComponent(crash_test_dummy.IX)
        rebuild.rebuild(crash_test_dummy, 0)
        newComponent = x.getComponent(crash_test_dummy.IX)
        newComponent.method()
        self.assertEqual(newComponent.__class__, crash_test_dummy.XA)
        from twisted.python import components
        self.assertRaises(ValueError, components.registerAdapter, crash_test_dummy.XA, crash_test_dummy.X, crash_test_dummy.IX)

    def test_UpdateInstance(self) -> None:
        global Foo, Buz
        b = Buz()

        class Foo:

            def foo(self) -> None:
                """
                Dummy method
                """

        class Buz(Bar, Baz):
            x = 10
        rebuild.updateInstance(b)
        assert hasattr(b, 'foo'), 'Missing method on rebuilt instance'
        assert hasattr(b, 'x'), 'Missing class attribute on rebuilt instance'

    def test_BananaInteraction(self) -> None:
        from twisted.python import rebuild
        from twisted.spread import banana
        rebuild.latestClass(banana.Banana)

    def test_hashException(self) -> None:
        """
        Rebuilding something that has a __hash__ that raises a non-TypeError
        shouldn't cause rebuild to die.
        """
        global unhashableObject
        unhashableObject = HashRaisesRuntimeError()

        def _cleanup() -> None:
            global unhashableObject
            unhashableObject = None
        self.addCleanup(_cleanup)
        rebuild.rebuild(rebuild)
        self.assertTrue(unhashableObject.hashCalled)

    def test_Sensitive(self) -> None:
        """
        L{twisted.python.rebuild.Sensitive}
        """
        from twisted.python import rebuild
        from twisted.python.rebuild import Sensitive

        class TestSensitive(Sensitive):

            def test_method(self) -> None:
                """
                Dummy method
                """
        testSensitive = TestSensitive()
        testSensitive.rebuildUpToDate()
        self.assertFalse(testSensitive.needRebuildUpdate())
        newException = rebuild.latestClass(Exception)
        self.assertEqual(repr(Exception), repr(newException))
        self.assertEqual(newException, testSensitive.latestVersionOf(newException))
        self.assertEqual(TestSensitive.test_method, testSensitive.latestVersionOf(TestSensitive.test_method))
        self.assertEqual(testSensitive.test_method, testSensitive.latestVersionOf(testSensitive.test_method))
        self.assertEqual(TestSensitive, testSensitive.latestVersionOf(TestSensitive))

        def myFunction() -> None:
            """
            Dummy method
            """
        self.assertEqual(myFunction, testSensitive.latestVersionOf(myFunction))