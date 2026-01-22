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
class ComputeTensorMethod:
    target: Literal[Target.DECLARATION, Target.DEFINITION]
    static_dispatch_backend_indices: List[BackendIndex]

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if Variant.method not in f.variants:
            return None
        assert not f.func.is_out_fn()
        assert f.func.arguments.self_arg is not None
        sig_group = CppSignatureGroup.from_native_function(f, method=True, fallback_binding=f.manual_cpp_binding)
        if self.target is Target.DECLARATION:
            result = ''
            for sig in sig_group.signatures():
                result += f'{sig.decl()} const;\n'
            return result
        if self.target is not Target.DEFINITION:
            assert_never(self.target)
        result = ''
        for sig in sig_group.signatures():
            target_sig = DispatcherSignature.from_schema(f.func)
            exprs = translate(sig.arguments(), target_sig.arguments(), method=True)
            exprs_str = ', '.join([e.expr for e in exprs])
            result += f'\n// aten::{f.func}\ninline {sig.defn(prefix='Tensor::')} const {{\n    return at::_ops::{f.func.name.unambiguous_name()}::call({exprs_str});\n}}\n'
        return result