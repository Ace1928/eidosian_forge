import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
class NotifierTests(unittest.TestCase):

    def setUp(self):
        obj = self.obj = NotifierTraits()
        obj.value1 = 0
        obj.value2 = 0
        obj.value1_count = 0
        obj.value2_count = 0

    def tearDown(self):
        obj = self.obj
        obj.on_trait_change(self.on_value1_changed, 'value1', remove=True)
        obj.on_trait_change(self.on_value2_changed, 'value2', remove=True)
        obj.on_trait_change(self.on_anytrait_changed, remove=True)

    def on_anytrait_changed(self, object, trait_name, old, new):
        if trait_name == 'value1':
            self.obj.value1_count += 1
        elif trait_name == 'value2':
            self.obj.value2_count += 1

    def on_value1_changed(self):
        self.obj.value1_count += 1

    def on_value2_changed(self):
        self.obj.value2_count += 1

    def test_simple(self):
        obj = self.obj
        obj.value1 = 1
        self.assertEqual(obj.value1_count, 2)
        self.assertEqual(obj.value2_count, 0)
        obj.value2 = 1
        self.assertEqual(obj.value1_count, 2)
        self.assertEqual(obj.value2_count, 2)

    def test_complex(self):
        obj = self.obj
        obj.on_trait_change(self.on_value1_changed, 'value1')
        obj.value1 = 1
        self.assertEqual(obj.value1_count, 3)
        self.assertEqual(obj.value2_count, 0)
        obj.on_trait_change(self.on_value2_changed, 'value2')
        obj.value2 = 1
        self.assertEqual(obj.value1_count, 3)
        self.assertEqual(obj.value2_count, 3)
        obj.on_trait_change(self.on_anytrait_changed)
        obj.value1 = 2
        self.assertEqual(obj.value1_count, 7)
        self.assertEqual(obj.value2_count, 3)
        obj.value1 = 2
        self.assertEqual(obj.value1_count, 7)
        self.assertEqual(obj.value2_count, 3)
        obj.value2 = 2
        self.assertEqual(obj.value1_count, 7)
        self.assertEqual(obj.value2_count, 7)
        obj.on_trait_change(self.on_value1_changed, 'value1', remove=True)
        obj.value1 = 3
        self.assertEqual(obj.value1_count, 10)
        self.assertEqual(obj.value2_count, 7)
        obj.on_trait_change(self.on_value2_changed, 'value2', remove=True)
        obj.value2 = 3
        self.assertEqual(obj.value1_count, 10)
        self.assertEqual(obj.value2_count, 10)
        obj.on_trait_change(self.on_anytrait_changed, remove=True)
        obj.value1 = 4
        self.assertEqual(obj.value1_count, 12)
        self.assertEqual(obj.value2_count, 10)
        obj.value2 = 4
        self.assertEqual(obj.value1_count, 12)
        self.assertEqual(obj.value2_count, 12)