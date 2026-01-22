import inspect
import os
import sys
import unittest
from collections.abc import Sequence
from typing import List
from bpython import inspection
from bpython.test.fodder import encoding_ascii
from bpython.test.fodder import encoding_latin1
from bpython.test.fodder import encoding_utf8
class TestSafeGetAttribute(unittest.TestCase):

    def test_lookup_on_object(self):
        a = A()
        a.x = 1
        self.assertEqual(inspection.getattr_safe(a, 'x'), 1)
        self.assertEqual(inspection.getattr_safe(a, 'a'), 'a')
        b = B()
        b.y = 2
        self.assertEqual(inspection.getattr_safe(b, 'y'), 2)
        self.assertEqual(inspection.getattr_safe(b, 'a'), 'a')
        self.assertEqual(inspection.getattr_safe(b, 'b'), 'b')
        self.assertEqual(inspection.hasattr_safe(b, 'y'), True)
        self.assertEqual(inspection.hasattr_safe(b, 'b'), True)

    def test_avoid_running_properties(self):
        p = Property()
        self.assertEqual(inspection.getattr_safe(p, 'prop'), Property.prop)
        self.assertEqual(inspection.hasattr_safe(p, 'prop'), True)

    def test_lookup_with_slots(self):
        s = Slots()
        s.s1 = 's1'
        self.assertEqual(inspection.getattr_safe(s, 's1'), 's1')
        with self.assertRaises(AttributeError):
            inspection.getattr_safe(s, 's2')
        self.assertEqual(inspection.hasattr_safe(s, 's1'), True)
        self.assertEqual(inspection.hasattr_safe(s, 's2'), False)

    def test_lookup_on_slots_classes(self):
        sga = inspection.getattr_safe
        s = SlotsSubclass()
        self.assertIsInstance(sga(Slots, 's1'), member_descriptor)
        self.assertIsInstance(sga(SlotsSubclass, 's1'), member_descriptor)
        self.assertIsInstance(sga(SlotsSubclass, 's4'), property)
        self.assertIsInstance(sga(s, 's4'), property)
        self.assertEqual(inspection.hasattr_safe(s, 's1'), False)
        self.assertEqual(inspection.hasattr_safe(s, 's4'), True)

    def test_lookup_on_overridden_methods(self):
        sga = inspection.getattr_safe
        self.assertEqual(sga(OverriddenGetattr(), 'a'), 1)
        self.assertEqual(sga(OverriddenGetattribute(), 'a'), 1)
        self.assertEqual(sga(OverriddenMRO(), 'a'), 1)
        with self.assertRaises(AttributeError):
            sga(OverriddenGetattr(), 'b')
        with self.assertRaises(AttributeError):
            sga(OverriddenGetattribute(), 'b')
        with self.assertRaises(AttributeError):
            sga(OverriddenMRO(), 'b')
        self.assertEqual(inspection.hasattr_safe(OverriddenGetattr(), 'b'), False)
        self.assertEqual(inspection.hasattr_safe(OverriddenGetattribute(), 'b'), False)
        self.assertEqual(inspection.hasattr_safe(OverriddenMRO(), 'b'), False)