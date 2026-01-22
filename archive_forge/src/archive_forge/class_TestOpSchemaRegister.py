import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
@parameterized.parameterized_class([{'op_type': 'CustomOp', 'op_version': 5, 'op_domain': '', 'trap_op_version': [1, 2, 6, 7]}, {'op_type': 'CustomOp', 'op_version': 5, 'op_domain': 'test', 'trap_op_version': [1, 2, 6, 7]}])
class TestOpSchemaRegister(unittest.TestCase):
    op_type: str
    op_version: int
    op_domain: str
    trap_op_version: List[int]

    def setUp(self) -> None:
        self.assertFalse(onnx.defs.has(self.op_type, self.op_domain))

    def tearDown(self) -> None:
        for version in [*self.trap_op_version, self.op_version]:
            with contextlib.suppress(onnx.defs.SchemaError):
                onnx.defs.deregister_schema(self.op_type, version, self.op_domain)

    def test_register_multi_schema(self):
        for version in [*self.trap_op_version, self.op_version]:
            op_schema = defs.OpSchema(self.op_type, self.op_domain, version)
            onnx.defs.register_schema(op_schema)
            self.assertTrue(onnx.defs.has(self.op_type, version, self.op_domain))
        for version in [*self.trap_op_version, self.op_version]:
            registered_op = onnx.defs.get_schema(op_schema.name, version, op_schema.domain)
            op_schema = defs.OpSchema(self.op_type, self.op_domain, version)
            self.assertEqual(str(registered_op), str(op_schema))

    def test_using_the_specified_version_in_onnx_check(self):
        input = f'\n            <\n                ir_version: 7,\n                opset_import: [\n                    "{self.op_domain}" : {self.op_version}\n                ]\n            >\n            agraph (float[N, 128] X, int32 Y) => (float[N] Z)\n            {{\n                Z = {self.op_domain}.{self.op_type}<attr1=[1,2]>(X, Y)\n            }}\n           '
        model = onnx.parser.parse_model(input)
        op_schema = defs.OpSchema(self.op_type, self.op_domain, self.op_version, inputs=[defs.OpSchema.FormalParameter('input1', 'T'), defs.OpSchema.FormalParameter('input2', 'int32')], outputs=[defs.OpSchema.FormalParameter('output1', 'T')], type_constraints=[('T', ['tensor(float)'], '')], attributes=[defs.OpSchema.Attribute('attr1', defs.OpSchema.AttrType.INTS, 'attr1 description')])
        with self.assertRaises(onnx.checker.ValidationError):
            onnx.checker.check_model(model, check_custom_domain=True)
        onnx.defs.register_schema(op_schema)
        for version in self.trap_op_version:
            onnx.defs.register_schema(defs.OpSchema(self.op_type, self.op_domain, version, outputs=[defs.OpSchema.FormalParameter('output1', 'int32')]))
        onnx.checker.check_model(model, check_custom_domain=True)

    def test_register_schema_raises_error_when_registering_a_schema_twice(self):
        op_schema = defs.OpSchema(self.op_type, self.op_domain, self.op_version)
        onnx.defs.register_schema(op_schema)
        with self.assertRaises(onnx.defs.SchemaError):
            onnx.defs.register_schema(op_schema)

    def test_deregister_the_specified_schema(self):
        for version in [*self.trap_op_version, self.op_version]:
            op_schema = defs.OpSchema(self.op_type, self.op_domain, version)
            onnx.defs.register_schema(op_schema)
            self.assertTrue(onnx.defs.has(op_schema.name, version, op_schema.domain))
        onnx.defs.deregister_schema(op_schema.name, self.op_version, op_schema.domain)
        for version in self.trap_op_version:
            self.assertTrue(onnx.defs.has(op_schema.name, version, op_schema.domain))
        if onnx.defs.has(op_schema.name, self.op_version, op_schema.domain):
            schema = onnx.defs.get_schema(op_schema.name, self.op_version, op_schema.domain)
            self.assertLess(schema.since_version, self.op_version)

    def test_deregister_schema_raises_error_when_opschema_does_not_exist(self):
        with self.assertRaises(onnx.defs.SchemaError):
            onnx.defs.deregister_schema(self.op_type, self.op_version, self.op_domain)

    def test_legacy_schema_accessible_after_deregister(self):
        op_schema = defs.OpSchema(self.op_type, self.op_domain, self.op_version)
        onnx.defs.register_schema(op_schema)
        schema_a = onnx.defs.get_schema(op_schema.name, op_schema.since_version, op_schema.domain)
        schema_b = onnx.defs.get_schema(op_schema.name, op_schema.domain)

        def filter_schema(schemas):
            return [op for op in schemas if op.name == op_schema.name]
        schema_c = filter_schema(onnx.defs.get_all_schemas())
        schema_d = filter_schema(onnx.defs.get_all_schemas_with_history())
        self.assertEqual(len(schema_c), 1)
        self.assertEqual(len(schema_d), 1)
        self.assertEqual(str(schema_a), str(op_schema))
        self.assertEqual(str(schema_b), str(op_schema))
        self.assertEqual(str(schema_c[0]), str(op_schema))
        self.assertEqual(str(schema_d[0]), str(op_schema))