import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
def test_register_multi_schema(self):
    for version in [*self.trap_op_version, self.op_version]:
        op_schema = defs.OpSchema(self.op_type, self.op_domain, version)
        onnx.defs.register_schema(op_schema)
        self.assertTrue(onnx.defs.has(self.op_type, version, self.op_domain))
    for version in [*self.trap_op_version, self.op_version]:
        registered_op = onnx.defs.get_schema(op_schema.name, version, op_schema.domain)
        op_schema = defs.OpSchema(self.op_type, self.op_domain, version)
        self.assertEqual(str(registered_op), str(op_schema))