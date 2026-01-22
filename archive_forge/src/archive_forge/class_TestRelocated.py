import sys
import pyomo.common.unittest as unittest
from pyomo.common import DeveloperError
from pyomo.common.deprecation import (
from pyomo.common.log import LoggingIntercept
from io import StringIO
import logging
class TestRelocated(unittest.TestCase):

    def test_relocated_class(self):
        warning = "DEPRECATED: the 'myFoo' class has been moved to 'pyomo.common.tests.relocated.Bar'"
        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo'):
            from pyomo.common.tests.test_deprecated import myFoo
        self.assertEqual(myFoo.data, 42)
        self.assertIn(warning, OUT.getvalue().replace('\n', ' '))
        from pyomo.common.tests import relocated
        self.assertNotIn('Foo', dir(relocated))
        self.assertNotIn('Foo_2', dir(relocated))
        warning = "DEPRECATED: the 'Foo_2' class has been moved to 'pyomo.common.tests.relocated.Bar'"
        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo'):
            self.assertIs(relocated.Foo_2, relocated.Bar)
            self.assertEqual(relocated.Foo_2.data, 42)
        self.assertIn(warning, OUT.getvalue().replace('\n', ' '))
        self.assertNotIn('Foo', dir(relocated))
        self.assertIn('Foo_2', dir(relocated))
        self.assertIs(relocated.Foo_2, relocated.Bar)
        warning = "DEPRECATED: the 'Foo' class has been moved to 'pyomo.common.tests.test_deprecated.Bar'"
        OUT = StringIO()
        with LoggingIntercept(OUT, 'pyomo'):
            from pyomo.common.tests.relocated import Foo
            self.assertEqual(Foo.data, 21)
        self.assertIn(warning, OUT.getvalue().replace('\n', ' '))
        self.assertIn('Foo', dir(relocated))
        self.assertIn('Foo_2', dir(relocated))
        self.assertIs(relocated.Foo, Bar)
        with self.assertRaisesRegex(AttributeError, "(?:module 'pyomo.common.tests.relocated')|(?:'module' object) has no attribute 'Baz'"):
            relocated.Baz.data
        self.assertEqual(relocated.Foo_3, '_3')
        with self.assertRaisesRegex(AttributeError, "(?:module 'pyomo.common.tests.test_deprecated')|(?:'module' object) has no attribute 'Baz'"):
            sys.modules[__name__].Baz.data

    def test_relocated_message(self):
        with LoggingIntercept() as LOG:
            self.assertIs(_import_object('oldName', 'pyomo.common.tests.test_deprecated.logger', 'TBD', None, None), logger)
        self.assertRegex(LOG.getvalue().replace('\n', ' '), "DEPRECATED: the 'oldName' attribute has been moved to 'pyomo.common.tests.test_deprecated.logger'")
        with LoggingIntercept() as LOG:
            self.assertIs(_import_object('oldName', 'pyomo.common.tests.test_deprecated._import_object', 'TBD', None, None), _import_object)
        self.assertRegex(LOG.getvalue().replace('\n', ' '), "DEPRECATED: the 'oldName' function has been moved to 'pyomo.common.tests.test_deprecated._import_object'")
        with LoggingIntercept() as LOG:
            self.assertIs(_import_object('oldName', 'pyomo.common.tests.test_deprecated.TestRelocated', 'TBD', None, None), TestRelocated)
        self.assertRegex(LOG.getvalue().replace('\n', ' '), "DEPRECATED: the 'oldName' class has been moved to 'pyomo.common.tests.test_deprecated.TestRelocated'")

    def test_relocated_module(self):
        with LoggingIntercept() as LOG:
            from pyomo.common.tests.relo_mod import ReloClass
        self.assertRegex(LOG.getvalue().replace('\n', ' '), "DEPRECATED: The 'pyomo\\.common\\.tests\\.relo_mod' module has been moved to 'pyomo\\.common\\.tests\\.relo_mod_new'. Please update your import. \\(deprecated in 1\\.2\\) \\(called from .*test_deprecated\\.py")
        with LoggingIntercept() as LOG:
            import pyomo.common.tests.relo_mod as relo
        self.assertEqual(LOG.getvalue(), '')
        import pyomo.common.tests.relo_mod_new as relo_new
        self.assertIs(relo, relo_new)
        self.assertEqual(relo.RELO_ATTR, 42)
        self.assertIs(ReloClass, relo_new.ReloClass)