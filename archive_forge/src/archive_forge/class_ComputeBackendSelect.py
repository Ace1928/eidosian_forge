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
class ComputeBackendSelect:
    target: Literal[Target.DEFINITION, Target.REGISTRATION]
    selector: SelectiveBuilder

    @method_with_native_function
    def __call__(self, f: NativeFunction) -> Optional[str]:
        if not needs_backend_select(f, self.selector):
            return None
        name = native.name(f.func)
        native_sig = NativeSignature(f.func, symint=True)
        native_tensor_args = [a for a in native_sig.arguments() if isinstance(a.argument, Argument) and a.argument.type.is_tensor_like()]
        dispatcher_sig = DispatcherSignature.from_schema(f.func)
        sig: Union[NativeSignature, DispatcherSignature]
        sig = dispatcher_sig
        dispatcher_exprs = dispatcher_sig.exprs()
        dispatch_key = 'c10::computeDispatchKey(dtype, layout, device)'
        if self.target is Target.DEFINITION:
            if native_tensor_args:
                assert f.func.arguments.has_tensor_arg()
                tensor_args = ', '.join((a.name for a in native_tensor_args))
                compute_dk = f'DispatchKeySet _dk_set = c10::DispatchKeySet({dispatch_key}) | c10::detail::multi_dispatch_key_set({tensor_args});\nDispatchKeySet _dk_mask = c10::DispatchKeySet(DispatchKeySet::FULL_AFTER, DispatchKey::BackendSelect);\nDispatchKeySet _dk = c10::impl::computeDispatchKeySet(_dk_set, _dk_mask);'
            else:
                assert not f.func.arguments.has_tensor_arg()
                compute_dk = f'DispatchKeySet _dk = c10::DispatchKeySet({dispatch_key});'
            return f'// aten::{f.func}\nC10_ALWAYS_INLINE\n{sig.defn(name)} {{\n  {compute_dk}\n  return at::_ops::{f.func.name.unambiguous_name()}::redispatch(\n      _dk, {', '.join((a.expr for a in dispatcher_exprs))});\n}}\n'
        elif self.target is Target.REGISTRATION:
            return f'm.impl("aten::{f.func.name}", TORCH_FN({name}));'
        else:
            assert_never(self.target)