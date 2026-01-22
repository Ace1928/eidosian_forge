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
def _get_kwargs_func_exclusion_list() -> List[str]:
    return ['diagonal', 'round_', 'round', 'scatter_']