import contextlib
import unittest
from typing import List, Sequence
import parameterized
import onnx
from onnx import defs
class TestOpSchema(unittest.TestCase):

    def test_init(self):
        schema = defs.OpSchema('test_op', 'test_domain', 1)
        self.assertIsInstance(schema, defs.OpSchema)

    def test_init_with_inputs(self) -> None:
        op_schema = defs.OpSchema('test_op', 'test_domain', 1, inputs=[defs.OpSchema.FormalParameter('input1', 'T')], type_constraints=[('T', ['tensor(int64)'], '')])
        self.assertEqual(op_schema.name, 'test_op')
        self.assertEqual(op_schema.domain, 'test_domain')
        self.assertEqual(op_schema.since_version, 1)
        self.assertEqual(len(op_schema.inputs), 1)
        self.assertEqual(op_schema.inputs[0].name, 'input1')
        self.assertEqual(op_schema.inputs[0].type_str, 'T')
        self.assertEqual(len(op_schema.type_constraints), 1)
        self.assertEqual(op_schema.type_constraints[0].type_param_str, 'T')
        self.assertEqual(op_schema.type_constraints[0].allowed_type_strs, ['tensor(int64)'])

    def test_init_creates_multi_input_output_schema(self) -> None:
        op_schema = defs.OpSchema('test_op', 'test_domain', 1, inputs=[defs.OpSchema.FormalParameter('input1', 'T'), defs.OpSchema.FormalParameter('input2', 'T')], outputs=[defs.OpSchema.FormalParameter('output1', 'T'), defs.OpSchema.FormalParameter('output2', 'T')], type_constraints=[('T', ['tensor(int64)'], '')], attributes=[defs.OpSchema.Attribute('attr1', defs.OpSchema.AttrType.INTS, 'attr1 description')])
        self.assertEqual(len(op_schema.inputs), 2)
        self.assertEqual(op_schema.inputs[0].name, 'input1')
        self.assertEqual(op_schema.inputs[0].type_str, 'T')
        self.assertEqual(op_schema.inputs[1].name, 'input2')
        self.assertEqual(op_schema.inputs[1].type_str, 'T')
        self.assertEqual(len(op_schema.outputs), 2)
        self.assertEqual(op_schema.outputs[0].name, 'output1')
        self.assertEqual(op_schema.outputs[0].type_str, 'T')
        self.assertEqual(op_schema.outputs[1].name, 'output2')
        self.assertEqual(op_schema.outputs[1].type_str, 'T')
        self.assertEqual(len(op_schema.type_constraints), 1)
        self.assertEqual(op_schema.type_constraints[0].type_param_str, 'T')
        self.assertEqual(op_schema.type_constraints[0].allowed_type_strs, ['tensor(int64)'])
        self.assertEqual(len(op_schema.attributes), 1)
        self.assertEqual(op_schema.attributes['attr1'].name, 'attr1')
        self.assertEqual(op_schema.attributes['attr1'].type, defs.OpSchema.AttrType.INTS)
        self.assertEqual(op_schema.attributes['attr1'].description, 'attr1 description')

    def test_init_without_optional_arguments(self) -> None:
        op_schema = defs.OpSchema('test_op', 'test_domain', 1)
        self.assertEqual(op_schema.name, 'test_op')
        self.assertEqual(op_schema.domain, 'test_domain')
        self.assertEqual(op_schema.since_version, 1)
        self.assertEqual(len(op_schema.inputs), 0)
        self.assertEqual(len(op_schema.outputs), 0)
        self.assertEqual(len(op_schema.type_constraints), 0)

    def test_name(self):
        with self.assertRaises(TypeError):
            defs.OpSchema(domain='test_domain', since_version=1)
        with self.assertRaises(TypeError):
            defs.OpSchema(123, 'test_domain', 1)
        schema = defs.OpSchema('test_op', 'test_domain', 1)
        self.assertEqual(schema.name, 'test_op')

    def test_domain(self):
        with self.assertRaises(TypeError):
            defs.OpSchema(name='test_op', since_version=1)
        with self.assertRaises(TypeError):
            defs.OpSchema('test_op', 123, 1)
        schema = defs.OpSchema('test_op', 'test_domain', 1)
        self.assertEqual(schema.domain, 'test_domain')

    def test_since_version(self):
        with self.assertRaises(TypeError):
            defs.OpSchema('test_op', 'test_domain')
        schema = defs.OpSchema('test_op', 'test_domain', 1)
        self.assertEqual(schema.since_version, 1)

    def test_doc(self):
        schema = defs.OpSchema('test_op', 'test_domain', 1, doc='test_doc')
        self.assertEqual(schema.doc, 'test_doc')

    def test_inputs(self):
        inputs = [defs.OpSchema.FormalParameter(name='input1', type_str='T', description='The first input.')]
        schema = defs.OpSchema('test_op', 'test_domain', 1, inputs=inputs, type_constraints=[('T', ['tensor(int64)'], '')])
        self.assertEqual(len(schema.inputs), 1)
        self.assertEqual(schema.inputs[0].name, 'input1')
        self.assertEqual(schema.inputs[0].type_str, 'T')
        self.assertEqual(schema.inputs[0].description, 'The first input.')

    def test_outputs(self):
        outputs = [defs.OpSchema.FormalParameter(name='output1', type_str='T', description='The first output.')]
        schema = defs.OpSchema('test_op', 'test_domain', 1, outputs=outputs, type_constraints=[('T', ['tensor(int64)'], '')])
        self.assertEqual(len(schema.outputs), 1)
        self.assertEqual(schema.outputs[0].name, 'output1')
        self.assertEqual(schema.outputs[0].type_str, 'T')
        self.assertEqual(schema.outputs[0].description, 'The first output.')