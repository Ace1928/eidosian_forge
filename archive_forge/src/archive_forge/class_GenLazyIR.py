import itertools
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torchgen.api.dispatcher as dispatcher
from torchgen.api.lazy import (
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import method_with_native_function
from torchgen.dest.lazy_ts_lowering import ts_lowering_body
from torchgen.model import (
@dataclass(frozen=True)
class GenLazyIR(ABC):
    backend_index: BackendIndex
    backend_name: str
    node_base: str
    use_lazy_shape: bool

    @method_with_native_function
    def __call__(self, f: Union[NativeFunctionsGroup, NativeFunction]) -> List[str]:
        func = f.functional.func if isinstance(f, NativeFunctionsGroup) else f.func
        metadata = self.backend_index.get_kernel(f.functional if isinstance(f, NativeFunctionsGroup) else f)
        schema = LazyIrSchema(func, symint=metadata is not None and metadata.supports_symint())
        return self.gen(schema)

    def lowering_function(self, schema: LazyIrSchema) -> str:
        return ''

    def create_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        return ''

    def can_be_reused_function(self, schema: LazyIrSchema, node_ctor_args: str) -> str:
        return f'bool CanBeReused({node_ctor_args}) const {{\n    return false;\n    }}'

    def node_base_ctor_call(self, schema: LazyIrSchema) -> str:
        value_args = schema.filtered_args(values=True, scalars=False)
        base_ctor_value_args_list = []
        for arg in value_args:
            if isinstance(arg.lazy_type, (BaseCType, VectorCType)):
                base_ctor_value_args_list.append(f'{arg.name}')
            elif isinstance(arg.lazy_type, OptionalCType):
                base_ctor_value_args_list.append(f'{arg.name}.value_or(kNullValue)')
            else:
                raise AssertionError(f'Unsupported type ({arg.lazy_type}) - add support if necessary')
        base_ctor_value_args = ', '.join(base_ctor_value_args_list)
        scalar_args = schema.filtered_args(values=False, scalars=True)
        if schema.properties.ShapePrecompute:
            shape_ctor_arg = 'std::move(shapes),'
        elif schema.properties.ShapeCompute:
            shape_args = [a.name for a in value_args]
            shape_args.extend((a.name for a in scalar_args))
            shape_ctor_arg = f'compute_shape_{schema.name}({', '.join(shape_args)}),'
        elif schema.properties.ShapeCache:
            shape_args = [f'operand({i})' for i in range(len(value_args))]
            shape_args.extend((a.name for a in scalar_args))
            shape_ctor_arg = f'[&](){{ return compute_shape_{schema.name}({', '.join(shape_args)})[0]; }},'
        else:
            shape_ctor_arg = ''
        scalar_hashes = ', '.join((f'{a.name}' for a in scalar_args))
        return f'{self.node_base}(\n              {schema.node_name}::ClassOpKind(),\n              OpList{{{base_ctor_value_args}}},\n              {shape_ctor_arg}\n              /* num_outputs */ {len(schema.returns)},\n              torch::lazy::MHash({scalar_hashes}))'

    def gen(self, schema: LazyIrSchema) -> List[str]:
        opkind = schema.opkind or aten_symbol(schema)
        all_args = schema.filtered_args()
        value_args = schema.filtered_args(values=True, scalars=False)
        scalar_args = schema.filtered_args(values=False, scalars=True)
        ctor_args = [f'const {i.lazy_type.cpp_type()}& {i.name}' for i in all_args]
        reuse_ctor_args = ', '.join(ctor_args)
        if self.use_lazy_shape and schema.properties.ShapePrecompute:
            ctor_args.append('std::vector<torch::lazy::Shape>&& shapes')
        node_ctor_args = ', '.join(ctor_args)
        scalar_initializers = ',\n        '.join([f'{a.name}({a.name}.has_value() ? c10::make_optional(std::string(*{a.name})) : c10::nullopt)' if a.lazy_type.cpp_type() == 'c10::optional<c10::string_view>' else f'{a.name}({a.name})' for a in scalar_args])
        if len(scalar_initializers):
            scalar_initializers = f',\n        {scalar_initializers}'
        scalar_decls = '\n  '.join([f'std::string {a.name};' if a.lazy_type.cpp_type() == 'c10::string_view' else f'c10::optional<std::string> {a.name};' if a.lazy_type.cpp_type() == 'c10::optional<c10::string_view>' else f'{a.lazy_type.cpp_type()} {a.name};' for a in scalar_args])
        optional_values = [arg.name for arg in schema.filtered_args(values=True, scalars=False) if isinstance(arg.lazy_type, OptionalCType)]
        has_optional_decls = '\n  '.join([f'bool has_{value}: 1;' for value in optional_values])
        has_optional_defs = '\n    '.join([f'has_{value} = !!{value};' for value in optional_values])
        members_to_string = []
        for arg in scalar_args:
            if isinstance(arg.lazy_type, OptionalCType):
                value = f'{arg.name}.value()'
                if arg.is_generator:
                    value = '"torch.Generator()"'
                members_to_string.append(f'if ({arg.name}.has_value()) {{\n      ss << ", {arg.name}=" << {value};\n    }} else {{\n      ss << ", {arg.name}=null";\n    }}')
            else:
                members_to_string.append(f'ss << ", {arg.name}=" << {arg.name};')
        members_to_string_str = '\n    '.join(members_to_string)
        return [f'class {schema.node_name} : public {self.node_base} {{\n public:\n  static torch::lazy::OpKind ClassOpKind() {{\n    return torch::lazy::OpKind({opkind});\n  }}\n\n  {schema.node_name}({node_ctor_args})\n      : {self.node_base_ctor_call(schema)}{scalar_initializers}\n  {{\n    {has_optional_defs}\n  }}\n\n  std::string ToString() const override {{\n    std::stringstream ss;\n    ss << {self.node_base}::ToString();\n    {members_to_string_str}\n    return ss.str();\n  }}\n\n  {self.create_function(schema, reuse_ctor_args)}\n\n  {self.can_be_reused_function(schema, reuse_ctor_args)}\n\n  {self.lowering_function(schema)}\n\n  {scalar_decls}\n  {has_optional_decls}\n\n}};\n\n']