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
def _compared_saved_loaded(self, m):

    def extract_files(buffer):
        archive = zipfile.ZipFile(buffer)
        self.assertEqual(len(set(archive.namelist())), len(archive.namelist()))
        files = list(filter(lambda x: x.startswith('archive/code/'), archive.namelist()))
        code_files_str = filter(lambda x: x.endswith('.py'), files)
        code_files_stream = (archive.open(f) for f in code_files_str)
        code_files = (''.join([line.decode() for line in file]) for file in code_files_stream)
        debug_files_str = filter(lambda f: f.endswith('.debug_pkl'), files)
        debug_files_stream = (archive.open(f) for f in debug_files_str)
        debug_files = (pickle.load(f) for f in debug_files_stream)
        return (code_files, debug_files)
    with torch._jit_internal._disable_emit_hooks():
        try:
            if len(m.code) == 0:
                return
            if isinstance(m, torch._C.ScriptModule):
                if len(m._method_names()) == 0:
                    return
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)
            buffer_copy = buffer.getvalue()
            code_files, debug_files = extract_files(buffer)
        except RuntimeError as e:
            if not self._isHookExceptionOk(e):
                raise
            else:
                return
        buffer2 = io.BytesIO(buffer_copy)
        imported = torch.jit.load(buffer2)
        saved_module_buffer_2 = io.BytesIO()
        torch.jit.save(imported, saved_module_buffer_2)
        saved_module_buffer_2.seek(0)
        code_files_2, debug_files_2 = extract_files(saved_module_buffer_2)
        for a, b in zip(code_files, code_files_2):
            self.assertMultiLineEqual(a, b)
        if isinstance(m, torch._C.ScriptModule):
            self.assertTrue(torch._C._ivalue_tags_match(m, imported._c))