import json
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union
import torchgen.api.cpp as cpp
from torchgen.context import native_function_manager
from torchgen.model import (
from torchgen.static_runtime import config
class GenOpTestCase:

    def out_variant(self, groups: Sequence[NativeFunctionsGroup]) -> str:
        if not groups:
            return ''
        generated_type_variants = []
        for g in groups:
            with native_function_manager(g):
                assert is_supported(g)
                assert isinstance(g, NativeFunctionsGroup)
                generated_type_variant = self.out_variant_op_test_case_generator(g)
                generated_type_variants.append(generated_type_variant)
        return '\n'.join(generated_type_variants)

    def view(self, groups: Sequence[NativeFunctionsViewGroup]) -> str:
        if not groups:
            return ''
        generated_type_variants = []
        for g in groups:
            with native_function_manager(g):
                assert is_supported(g)
                assert isinstance(g, NativeFunctionsViewGroup)
                generated_type_variant = self.view_op_test_case_generator(g)
                generated_type_variants.append(generated_type_variant)
        return '\n'.join(generated_type_variants)

    def out_variant_op_test_case_generator(self, g: NativeFunctionsGroup) -> str:
        schema = g.functional.func
        schema_str = str(schema)
        assert schema_str.find('(') > 0
        type_variant_op_name = schema_str[:schema_str.find('(')].replace('.', '_')
        op_name = op_name_from_group(g)
        assert type_variant_op_name.startswith(op_name)
        arg_types = generate_test_ir_arguments(schema)
        arg_declarations = ', '.join((arg_name if arg_type is None else f'{arg_name}: {arg_type}' for arg_name, arg_type in arg_types))
        arg_names = ', '.join((arg_name for arg_name, _ in arg_types))
        assert len(schema.returns) == 1 and isinstance(schema.returns[0].type, BaseType) and (schema.returns[0].type.name is BaseTy.Tensor)
        test_value_definitions = generate_test_value_definitions(schema, 0)
        test_value_names = generate_test_value_names(schema, 0)
        test_value_definitions2 = generate_test_value_definitions(schema, 1)
        test_value_names2 = generate_test_value_names(schema, 1)
        check_resize = 'true' if should_check_resize(schema) else 'false'
        generated = f'\nTEST(StaticRuntime, autogen_{type_variant_op_name}) {{\n  const std::string script = R"IR(\n    graph({arg_declarations}):\n        %bias: None = prim::Constant()\n        %ret = aten::{op_name}({arg_names})\n        %cloned = aten::clone(%ret, %bias)\n        return (%cloned)\n  )IR";\n\n  {test_value_definitions}\n  std::vector<IValue> args{{{test_value_names}}};\n  testStaticRuntime(script, args, {{}}, /*use_allclose=*/false, /*use_equalnan=*/false, /*check_resize=*/{check_resize});\n\n  {test_value_definitions2}\n  std::vector<IValue> args2{{{test_value_names2}}};\n  testStaticRuntime(script, args, args2, /*use_allclose=*/false, /*use_equalnan=*/false, /*check_resize=*/{check_resize});\n\n}}\n'
        return generated

    def view_op_test_case_generator(self, g: NativeFunctionsViewGroup) -> str:
        schema = g.view.func
        schema_str = str(schema)
        assert schema_str.find('(') > 0
        type_variant_op_name = schema_str[:schema_str.find('(')].replace('.', '_')
        op_name = g.view.root_name
        assert type_variant_op_name.startswith(op_name)
        arg_types = generate_test_ir_arguments(schema)
        arg_declarations = ', '.join((arg_name if arg_type is None else f'{arg_name}: {arg_type}' for arg_name, arg_type in arg_types))
        arg_names = ', '.join((arg_name for arg_name, _ in arg_types))
        assert len(schema.returns) == 1 and isinstance(schema.returns[0].type, BaseType) and (schema.returns[0].type.name is BaseTy.Tensor)
        test_value_definitions = generate_test_value_definitions(schema, 0)
        test_value_names = generate_test_value_names(schema, 0)
        generated = f'\nTEST(StaticRuntime, autogen_{type_variant_op_name}) {{\n  const std::string script = R"IR(\n    graph({arg_declarations}):\n        %bias: None = prim::Constant()\n        %ret = aten::{op_name}({arg_names})\n        %cloned = aten::clone(%ret, %bias)\n        return (%cloned)\n  )IR";\n\n  {test_value_definitions}\n  std::vector<IValue> args{{{test_value_names}}};\n  testStaticRuntime(script, args);\n}}\n'
        return generated