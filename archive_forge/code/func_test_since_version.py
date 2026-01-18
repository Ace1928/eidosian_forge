import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def test_since_version(self):
    with self.assertRaises(TypeError):
        defs.OpSchema('test_op', 'test_domain')
    schema = defs.OpSchema('test_op', 'test_domain', 1)
    self.assertEqual(schema.since_version, 1)