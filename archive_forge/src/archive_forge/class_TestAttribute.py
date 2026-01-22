import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
class TestAttribute(unittest.TestCase):

    def test_init(self):
        name = 'test_attr'
        type_ = defs.OpSchema.AttrType.STRINGS
        description = 'Test attribute'
        attribute = defs.OpSchema.Attribute(name, type_, description)
        self.assertEqual(attribute.name, name)
        self.assertEqual(attribute.type, type_)
        self.assertEqual(attribute.description, description)

    def test_init_with_default_value(self):
        default_value = defs.get_schema('BatchNormalization').attributes['epsilon'].default_value
        self.assertIsInstance(default_value, onnx.AttributeProto)
        attribute = defs.OpSchema.Attribute('attr1', default_value, 'attr1 description')
        self.assertEqual(default_value, attribute.default_value)
        self.assertEqual('attr1', attribute.name)
        self.assertEqual('attr1 description', attribute.description)