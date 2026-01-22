import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
class TestTypeConstraintParam(unittest.TestCase):

    @parameterized.parameterized.expand([('single_type', 'T', ['tensor(float)'], 'Test description'), ('double_types', 'T', ['tensor(float)', 'tensor(int64)'], 'Test description'), ('tuple', 'T', ('tensor(float)', 'tensor(int64)'), 'Test description')])
    def test_init(self, _: str, type_param_str: str, allowed_types: Sequence[str], description: str) -> None:
        type_constraint = defs.OpSchema.TypeConstraintParam(type_param_str, allowed_types, description)
        self.assertEqual(type_constraint.description, description)
        self.assertEqual(type_constraint.allowed_type_strs, list(allowed_types))
        self.assertEqual(type_constraint.type_param_str, type_param_str)