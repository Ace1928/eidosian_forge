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

For procedural tests needed for __torch_function__, we use this function
to export method names and signatures as needed by the tests in
test/test_overrides.py.

python -m tools.autograd.gen_annotated_fn_args        aten/src/ATen/native/native_functions.yaml        aten/src/ATen/native/tags.yaml        $OUTPUT_DIR        tools/autograd

Where $OUTPUT_DIR is where you would like the files to be
generated.  In the full build system, OUTPUT_DIR is
torch/testing/_internal/generated
