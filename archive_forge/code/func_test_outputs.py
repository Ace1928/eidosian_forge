import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def test_outputs(self):
    outputs = [defs.OpSchema.FormalParameter(name='output1', type_str='T', description='The first output.')]
    schema = defs.OpSchema('test_op', 'test_domain', 1, outputs=outputs, type_constraints=[('T', ['tensor(int64)'], '')])
    self.assertEqual(len(schema.outputs), 1)
    self.assertEqual(schema.outputs[0].name, 'output1')
    self.assertEqual(schema.outputs[0].type_str, 'T')
    self.assertEqual(schema.outputs[0].description, 'The first output.')