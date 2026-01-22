import unittest
from zope.interface.tests import OptimizationTestMixin
class AdapterRegistryTests(VerifyingAdapterRegistryTests):

    def _getTargetClass(self):
        from zope.interface.adapter import AdapterRegistry
        return AdapterRegistry

    def test_ctor_no_bases(self):
        ar = self._makeOne()
        self.assertEqual(len(ar._v_subregistries), 0)

    def test_ctor_w_bases(self):
        base = self._makeOne()
        sub = self._makeOne([base])
        self.assertEqual(len(sub._v_subregistries), 0)
        self.assertEqual(len(base._v_subregistries), 1)
        self.assertIn(sub, base._v_subregistries)

    def test__setBases_removing_existing_subregistry(self):
        before = self._makeOne()
        after = self._makeOne()
        sub = self._makeOne([before])
        sub.__bases__ = [after]
        self.assertEqual(len(before._v_subregistries), 0)
        self.assertEqual(len(after._v_subregistries), 1)
        self.assertIn(sub, after._v_subregistries)

    def test__setBases_wo_stray_entry(self):
        before = self._makeOne()
        stray = self._makeOne()
        after = self._makeOne()
        sub = self._makeOne([before])
        sub.__dict__['__bases__'].append(stray)
        sub.__bases__ = [after]
        self.assertEqual(len(before._v_subregistries), 0)
        self.assertEqual(len(after._v_subregistries), 1)
        self.assertIn(sub, after._v_subregistries)

    def test__setBases_w_existing_entry_continuing(self):
        before = self._makeOne()
        after = self._makeOne()
        sub = self._makeOne([before])
        sub.__bases__ = [before, after]
        self.assertEqual(len(before._v_subregistries), 1)
        self.assertEqual(len(after._v_subregistries), 1)
        self.assertIn(sub, before._v_subregistries)
        self.assertIn(sub, after._v_subregistries)

    def test_changed_w_subregistries(self):
        base = self._makeOne()

        class Derived:
            _changed = None

            def changed(self, originally_changed):
                self._changed = originally_changed
        derived1, derived2 = (Derived(), Derived())
        base._addSubregistry(derived1)
        base._addSubregistry(derived2)
        orig = object()
        base.changed(orig)
        self.assertIs(derived1._changed, orig)
        self.assertIs(derived2._changed, orig)