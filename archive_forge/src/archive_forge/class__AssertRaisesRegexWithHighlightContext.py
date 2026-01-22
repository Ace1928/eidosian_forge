from torch.autograd import Variable
from torch.autograd.function import _nested_map
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
from torch.onnx import OperatorExportTypes
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
import zipfile
import functools
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_WINDOWS, \
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401
from contextlib import contextmanager
from functools import reduce
from io import StringIO
from collections import defaultdict
import importlib.util
import inspect
import io
import math
import os
import pickle
import sys
import tempfile
import textwrap
from importlib.abc import Loader
from typing import Any, Dict, List, Tuple, Union
class _AssertRaisesRegexWithHighlightContext:
    """
    A context manager that is useful for checking that error messages highlight
    the correct part of the source code.
    """

    def __init__(self, test_case, exception, regex, highlight):
        self.test_case = test_case
        self.exception_type = exception
        self.regex = regex
        self.highlight = highlight

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        with self.test_case.assertRaisesRegex(self.exception_type, self.regex):
            if type:
                raise value
        if self.highlight:
            FileCheck().check_source_highlighted(self.highlight).run(str(value))
        return True