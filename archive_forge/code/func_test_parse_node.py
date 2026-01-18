import unittest
from parameterized import parameterized
import onnx
from onnx import GraphProto, OperatorSetIdProto, checker
def test_parse_node(self):
    node = onnx.parser.parse_node('out1, out2 = SomeDomain.SomeOp <attr1 = 1> (in1, in2)')
    self.assertEqual(list(node.input), ['in1', 'in2'])
    self.assertEqual(list(node.output), ['out1', 'out2'])
    self.assertEqual(len(node.attribute), 1)
    attr_val = onnx.helper.get_node_attr_value(node, 'attr1')
    self.assertEqual(attr_val, 1)
    self.assertEqual(node.domain, 'SomeDomain')
    self.assertEqual(node.op_type, 'SomeOp')