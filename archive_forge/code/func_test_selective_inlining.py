import unittest
from onnx import inliner, parser
def test_selective_inlining(self):
    model = parser.parse_model('\n            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>\n            agraph (float[N] X) => (float[N] Y)\n            {\n                T = local.square (X)\n                Y = local.double_and_square (T)\n            }\n\n            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">\n            double_and_square (x) => (y) {\n                double = Add(x, x)\n                y = local.square(double)\n            }\n\n            <opset_import: [ "" : 17 ], domain: "local">\n            square (x) => (y) {\n                y = Mul (x, x)\n            }\n        ')
    inlined = inliner.inline_selected_functions(model, [('local', 'square')], exclude=False)
    inlined_nodes = inlined.graph.node
    self.assertEqual(len(inlined_nodes), 2)
    self.assertEqual(inlined_nodes[0].op_type, 'Mul')
    self.assertEqual(inlined_nodes[1].op_type, 'double_and_square')
    function_nodes = inlined.functions[0].node
    self.assertEqual(len(function_nodes), 2)
    self.assertEqual(function_nodes[0].op_type, 'Add')
    self.assertEqual(function_nodes[1].op_type, 'Mul')