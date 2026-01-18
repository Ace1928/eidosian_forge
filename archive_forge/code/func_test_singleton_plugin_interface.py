import gc
from io import StringIO
import weakref
from pyomo.common.unittest import TestCase
from pyomo.common.log import LoggingIntercept
from pyomo.common.plugin_base import (
def test_singleton_plugin_interface(self):

    class IFoo(Interface):
        pass

    class mySingleton(SingletonPlugin):
        implements(IFoo)
    ep = ExtensionPoint(IFoo)
    self.assertEqual(ep.extensions(), [])
    self.assertEqual(IFoo._plugins, {mySingleton: {0: (weakref.ref(mySingleton.__singleton__), False)}})
    self.assertIsNotNone(mySingleton.__singleton__)
    with self.assertRaisesRegex(RuntimeError, 'Cannot create multiple singleton plugin instances'):
        mySingleton()

    class myDerivedSingleton(mySingleton):
        pass
    self.assertEqual(ep.extensions(), [])
    self.assertEqual(IFoo._plugins, {mySingleton: {0: (weakref.ref(mySingleton.__singleton__), False)}, myDerivedSingleton: {1: (weakref.ref(myDerivedSingleton.__singleton__), False)}})
    self.assertIsNotNone(myDerivedSingleton.__singleton__)
    self.assertIsNot(mySingleton.__singleton__, myDerivedSingleton.__singleton__)

    class myDerivedNonSingleton(mySingleton):
        __singleton__ = False
    self.assertEqual(ep.extensions(), [])
    self.assertEqual(IFoo._plugins, {mySingleton: {0: (weakref.ref(mySingleton.__singleton__), False)}, myDerivedSingleton: {1: (weakref.ref(myDerivedSingleton.__singleton__), False)}, myDerivedNonSingleton: {}})
    self.assertIsNone(myDerivedNonSingleton.__singleton__)

    class myServiceSingleton(mySingleton):
        implements(IFoo, service=True)
    self.assertEqual(ep.extensions(), [myServiceSingleton.__singleton__])
    self.assertEqual(IFoo._plugins, {mySingleton: {0: (weakref.ref(mySingleton.__singleton__), False)}, myDerivedSingleton: {1: (weakref.ref(myDerivedSingleton.__singleton__), False)}, myDerivedNonSingleton: {}, myServiceSingleton: {2: (weakref.ref(myServiceSingleton.__singleton__), True)}})
    self.assertIsNotNone(myServiceSingleton.__singleton__)