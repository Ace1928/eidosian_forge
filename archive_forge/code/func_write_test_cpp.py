import argparse
import itertools
import os
from typing import Sequence, TypeVar, Union
from libfb.py.log import set_simple_logging  # type: ignore[import]
from torchgen import gen
from torchgen.context import native_function_manager
from torchgen.model import DispatchKey, NativeFunctionsGroup, NativeFunctionsViewGroup
from torchgen.static_runtime import config, generator
def write_test_cpp(cpp_ops: Sequence[str], file_path: str) -> None:
    code = '\n'.join(cpp_ops)
    generated = f'// @lint-ignore-every CLANGTIDY HOWTOEVEN\n// AUTO-GENERATED FROM: torchgen/static_runtime/gen_static_runtime_ops.py\n#include <gtest/gtest.h>\n#include <torch/csrc/jit/runtime/static/impl.h>\n#include <torch/torch.h>\n\n#include "test_utils.h"\n\nusing namespace caffe2;\nusing namespace torch;\nusing namespace torch::jit;\nusing namespace torch::jit::test;\nusing c10::IValue;\n\n{code}\n\n'
    with open(file_path, 'w') as f:
        f.write(generated)
    clang_format(file_path)