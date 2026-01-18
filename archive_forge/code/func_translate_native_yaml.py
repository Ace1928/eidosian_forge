import argparse
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple, Union
import yaml
from torchgen import dest
from torchgen.api import cpp as aten_cpp
from torchgen.api.types import CppSignature, CppSignatureGroup, CType, NamedCType
from torchgen.context import (
from torchgen.executorch.api import et_cpp
from torchgen.executorch.api.custom_ops import (
from torchgen.executorch.api.types import contextArg, ExecutorchCppSignature
from torchgen.executorch.api.unboxing import Unboxing
from torchgen.executorch.model import ETKernelIndex, ETKernelKey, ETParsedYaml
from torchgen.executorch.parse import ET_FIELDS, parse_et_yaml, parse_et_yaml_struct
from torchgen.gen import (
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
def translate_native_yaml(tags_yaml_path: str, aten_yaml_path: str, native_yaml_path: Optional[str], use_aten_lib: bool, out_file: TextIO) -> None:
    """Translates Executorch DSL dialect to use the same syntax as
    native_functions.yaml. The major difference is that Executorch DSL dialect
    supports "op" key, where it refers to the operator name in native_functions.yaml.

    For example, a functions.yaml may have the following entry:

    - op: add.out
      ...

    It needs to be translated to the following:

    - func: add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
      ...

    We go in aten_yaml_path and find the operator schema for "add.out" and add it
    to the original functions.yaml. We also add required field "variants", where for
    Executorch it will always be "function".

    For ATen mode we don't have to do the translation because native_yaml_path is
    the same as native_functions.yaml.

    Args:
        tags_yaml_path: Path to a tags.yaml file to satisfy codegen parsing.
            It is not optional.
        aten_yaml_path: Path to ATen operator yaml file native_functions.yaml.
        native_yaml_path: Path to a functions.yaml file to parse.
            If the path does not exist in the filesystem, it is treated as an
            empty file. If `custom_ops_yaml_path` exists, the contents of that
            file are appended to the yaml input to be parsed.
        use_aten_lib: We use this flag to determine if we want to generate native
            functions. In ATen mode we should generate out= variants.
        out_file: The IO object that we are writing into.
    Returns:
        None
    """
    if use_aten_lib:
        with open(aten_yaml_path) as aten_yaml:
            out_file.writelines(aten_yaml.readlines())
        return
    native_functions, persisted_fields = parse_et_yaml(aten_yaml_path, tags_yaml_path, None, skip_native_fns_gen=False)
    func_to_scoped_name: Dict[FunctionSchema, str] = {f.func: f'{f.namespace}::{f.func.name}' for f in native_functions}
    op_to_scoped_name: Dict[OperatorName, str] = {func.name: name for func, name in func_to_scoped_name.items()}
    schema_dict = {name: str(func) for func, name in func_to_scoped_name.items()}
    kernel_persist_dict: Dict[str, Dict[str, Any]] = {op_to_scoped_name[op]: v for op, v in persisted_fields.items()}
    if not native_yaml_path or not os.path.exists(native_yaml_path) or os.stat(native_yaml_path).st_size == 0:
        return
    with open(native_yaml_path) as native_yaml:
        native_es = yaml.load(native_yaml, Loader=LineLoader)
        if not native_es:
            return
        for e in native_es:
            assert isinstance(e.get('__line__'), int), e
            loc = Location(native_yaml_path, e.pop('__line__'))
            with context(lambda: f'in {loc}:\n  '):
                if 'variants' not in e:
                    e['variants'] = 'function'
                if 'func' in e:
                    continue
                assert isinstance(e.get('op'), str), e
                opname = e.pop('op')
                if '::' not in opname:
                    opname = 'aten::' + opname
                assert opname in schema_dict
                e['func'] = schema_dict.get(opname)
                if opname in kernel_persist_dict:
                    for k, v in kernel_persist_dict[opname].items():
                        e[k] = v
        yaml.dump(native_es, out_file, width=1000)