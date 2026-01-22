import argparse
import os
import pathlib
import re
from collections import Counter, namedtuple
from typing import (
import yaml
import torchgen.dest as dest
from torchgen.api.lazy import setValueT
from torchgen.api.types import BaseCppType
from torchgen.dest.lazy_ir import GenLazyIR, GenLazyNativeFuncDefinition, GenTSLazyIR
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import NativeFunction, NativeFunctionsGroup, OperatorName
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, FileManager, NamespaceHelper
from torchgen.yaml_utils import YamlLoader
from .gen_backend_stubs import (
class default_args:
    node_base: str = 'Node'
    node_base_hdr: Optional[str] = None
    shape_inference_hdr: str = 'torch/csrc/lazy/core/shape_inference.h'
    tensor_class: str = 'torch::lazy::LazyTensor'
    tensor_class_hdr: str = 'torch/csrc/lazy/core/tensor.h'
    lazy_ir_generator: Type[GenLazyIR] = GenLazyIR
    native_func_definition_generator: Type[GenLazyNativeFuncDefinition] = GenLazyNativeFuncDefinition
    backend_name: str = 'TorchScript'