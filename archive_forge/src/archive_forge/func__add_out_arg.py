import argparse
import os
import textwrap
from collections import defaultdict
from typing import Any, Dict, List, Sequence
import torchgen.api.python as python
from torchgen.context import with_native_function
from torchgen.gen import parse_native_yaml
from torchgen.model import Argument, BaseOperatorName, NativeFunction
from torchgen.utils import FileManager
from .gen_python_functions import (
def _add_out_arg(out_args: List[Dict[str, Any]], args: Sequence[Argument], *, is_kwarg_only: bool) -> None:
    for arg in args:
        if arg.default is not None:
            continue
        out_arg: Dict[str, Any] = {}
        out_arg['is_kwarg_only'] = str(is_kwarg_only)
        out_arg['name'] = arg.name
        out_arg['simple_type'] = python.argument_type_str(arg.type, simple_type=True)
        size_t = python.argument_type_size(arg.type)
        if size_t:
            out_arg['size'] = size_t
        out_args.append(out_arg)