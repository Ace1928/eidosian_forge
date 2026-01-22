import unittest
from traits.api import Delegate, HasTraits, Instance, Str
class DelegateTestCase(unittest.TestCase):
    """ Test cases for delegated traits. """

    def setUp(self):
        """ Reset all of the globals.
        """
        global baz_s_handler_self, baz_sd_handler_self, baz_u_handler_self
        global baz_t_handler_self, foo_s_handler_self, foo_t_handler_self
        baz_s_handler_self = None
        baz_sd_handler_self = None
        baz_u_handler_self = None
        baz_t_handler_self = None
        foo_s_handler_self = None
        foo_t_handler_self = None

    def test_reset(self):
        """ Test that a delegated trait may be reset.

        Deleting the attribute should reset the trait back to its initial
        delegation behavior.
        """
        f = Foo()
        b = Bar(foo=f)
        self.assertEqual(f.s, b.s)
        b.s = 'bar'
        self.assertNotEqual(f.s, b.s)
        del b.s
        self.assertEqual(f.s, b.s)

    def test_modify_prefix_handler_on_delegator(self):
        f = Foo()
        b = BazModify(foo=f)
        self.assertEqual(f.s, b.sd)
        b.sd = 'changed'
        self.assertEqual(f.s, b.sd)
        self.assertEqual(baz_s_handler_self, None)
        self.assertEqual(baz_sd_handler_self, b)

    def test_modify_prefix_handler_on_delegatee(self):
        f = Foo()
        b = BazModify(foo=f)
        self.assertEqual(f.s, b.sd)
        b.sd = 'changed'
        self.assertEqual(f.s, b.sd)
        self.assertEqual(foo_s_handler_self, f)

    def test_no_modify_prefix_handler_on_delegator(self):
        f = Foo()
        b = BazNoModify(foo=f)
        self.assertEqual(f.s, b.sd)
        b.sd = 'changed'
        self.assertNotEqual(f.s, b.sd)
        self.assertEqual(baz_s_handler_self, None)
        self.assertEqual(baz_sd_handler_self, b)

    def test_no_modify_prefix_handler_on_delegatee_not_called(self):
        f = Foo()
        b = BazNoModify(foo=f)
        self.assertEqual(f.s, b.sd)
        b.sd = 'changed'
        self.assertNotEqual(f.s, b.sd)
        self.assertEqual(foo_s_handler_self, None)

    def test_modify_handler_on_delegator(self):
        f = Foo()
        b = BazModify(foo=f)
        self.assertEqual(f.t, b.t)
        b.t = 'changed'
        self.assertEqual(f.t, b.t)
        self.assertEqual(baz_t_handler_self, b)

    def test_modify_handler_on_delegatee(self):
        f = Foo()
        b = BazModify(foo=f)
        self.assertEqual(f.t, b.t)
        b.t = 'changed'
        self.assertEqual(f.t, b.t)
        self.assertEqual(foo_t_handler_self, f)

    def test_no_modify_handler_on_delegator(self):
        f = Foo()
        b = BazNoModify(foo=f)
        self.assertEqual(f.t, b.t)
        b.t = 'changed'
        self.assertNotEqual(f.t, b.t)
        self.assertEqual(baz_t_handler_self, b)

    def test_no_modify_handler_on_delegatee_not_called(self):
        f = Foo()
        b = BazNoModify(foo=f)
        self.assertEqual(f.t, b.t)
        b.t = 'changed'
        self.assertNotEqual(f.t, b.t)
        self.assertEqual(foo_t_handler_self, None)

    def test_no_modify_handler_on_delegatee_direct_change(self):
        f = Foo()
        b = BazNoModify(foo=f)
        self.assertEqual(f.t, b.t)
        f.t = 'changed'
        self.assertEqual(f.t, b.t)
        self.assertEqual(foo_t_handler_self, f)

    def test_no_modify_handler_on_delegator_direct_change(self):
        f = Foo()
        b = BazNoModify(foo=f)
        self.assertEqual(f.t, b.t)
        f.t = 'changed'
        self.assertEqual(f.t, b.t)
        self.assertEqual(baz_t_handler_self, b)

    def test_modify_handler_on_delegatee_direct_change(self):
        f = Foo()
        b = BazModify(foo=f)
        self.assertEqual(f.t, b.t)
        f.t = 'changed'
        self.assertEqual(f.t, b.t)
        self.assertEqual(foo_t_handler_self, f)

    def test_modify_handler_on_delegator_direct_change(self):
        f = Foo()
        b = BazModify(foo=f)
        self.assertEqual(f.t, b.t)
        f.t = 'changed'
        self.assertEqual(f.t, b.t)
        self.assertEqual(baz_t_handler_self, b)

    def test_modify_handler_not_listenable(self):
        f = Foo()
        b = BazModify(foo=f)
        self.assertEqual(f.u, b.u)
        f.u = 'changed'
        self.assertEqual(f.u, b.u)
        self.assertEqual(baz_u_handler_self, None)

    def test_no_modify_handler_not_listenable(self):
        f = Foo()
        b = BazNoModify(foo=f)
        self.assertEqual(f.u, b.u)
        f.u = 'changed'
        self.assertEqual(f.u, b.u)
        self.assertEqual(baz_u_handler_self, None)