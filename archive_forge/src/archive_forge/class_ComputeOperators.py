import argparse
import functools
import json
import os
import pathlib
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from typing import (
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.gen_functionalization_type import (
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
from torchgen.yaml_utils import YamlDumper, YamlLoader
@dataclass(frozen=True)
class ComputeOperators:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: List[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> str:
        sig = DispatcherSignature.from_schema(f.func)
        name = f.func.name.unambiguous_name()
        if self.target is Target.DECLARATION:
            return f'\nstruct TORCH_API {name} {{\n  using schema = {sig.type()};\n  using ptr_schema = schema*;\n  // See Note [static constexpr char* members for windows NVCC]\n  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(name, "aten::{f.func.name.name}")\n  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(overload_name, "{f.func.name.overload_name}")\n  STATIC_CONSTEXPR_STR_INL_EXCEPT_WIN_CUDA(schema_str, {cpp_string(str(f.func))})\n  static {sig.defn(name='call', is_redispatching_fn=False)};\n  static {sig.defn(name='redispatch', is_redispatching_fn=True)};\n}};'
        elif self.target is Target.DEFINITION:
            defns = f'\nSTATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, name, "aten::{f.func.name.name}")\nSTATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, overload_name, "{f.func.name.overload_name}")\nSTATIC_CONST_STR_OUT_OF_LINE_FOR_WIN_CUDA({name}, schema_str, {cpp_string(str(f.func))})\n\n// aten::{f.func}\nstatic C10_NOINLINE c10::TypedOperatorHandle<{name}::schema> create_{name}_typed_handle() {{\n  return c10::Dispatcher::singleton()\n      .findSchemaOrThrow({name}::name, {name}::overload_name)\n      .typed<{name}::schema>();\n}}\n'
            for is_redispatching_fn in [False, True]:
                if is_redispatching_fn:
                    dispatcher_exprs_str = ', '.join(['dispatchKeySet'] + [a.name for a in sig.arguments()])
                    method_base = 'redispatch'
                else:
                    dispatcher_exprs_str = ', '.join([a.name for a in sig.arguments()])
                    method_base = 'call'
                dispatcher_call = method_base
                method_name = f'{name}::{method_base}'
                fn_body = f'\n    static auto op = create_{name}_typed_handle();\n    return op.{dispatcher_call}({dispatcher_exprs_str});'
                if not is_redispatching_fn and len(self.static_dispatch_backend_indices) > 0:
                    fn_body = static_dispatch(sig, f, backend_indices=self.static_dispatch_backend_indices)
                defns += f'\n// aten::{f.func}\n{sig.defn(name=method_name, is_redispatching_fn=is_redispatching_fn)} {{\n    {fn_body}\n}}\n'
            return defns
        else:
            assert_never(self.target)