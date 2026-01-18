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
def validate_shape_inference_header(shape_inference_hdr: str, expected_shape_infr_decls: List[str]) -> None:
    try:
        with open(shape_inference_hdr) as f:
            shape_infr_decls = f.read()
            shape_infr_decl_lines = set(shape_infr_decls.split('\n'))
    except OSError as e:
        raise AssertionError(f'Unable to read from the specified shape_inference_hdr file: {shape_inference_hdr}') from e
    shape_infr_regex = 'compute_shape_(\\w+)'
    actual_shape_infr_name_counts = Counter(re.findall(shape_infr_regex, shape_infr_decls))
    missing_decls = [decl for decl in expected_shape_infr_decls if decl not in shape_infr_decl_lines]
    if missing_decls:
        raise Exception(f'Missing shape inference function.\n\nPlease add declare this function in {shape_inference_hdr}:\n\nand implement it in the corresponding shape_inference.cpp file.\n\n{os.linesep.join(missing_decls)}')