import sys
import sysconfig
import subprocess
import pkgutil
import types
import importlib
import warnings
import numpy as np
import numpy
import pytest
from numpy.testing import IS_WASM
def test_numpy_namespace():
    undocumented = {'_add_newdoc_ufunc': 'numpy.core._multiarray_umath._add_newdoc_ufunc', 'add_docstring': 'numpy.core._multiarray_umath.add_docstring', 'add_newdoc': 'numpy.core.function_base.add_newdoc', 'add_newdoc_ufunc': 'numpy.core._multiarray_umath._add_newdoc_ufunc', 'byte_bounds': 'numpy.lib.utils.byte_bounds', 'compare_chararrays': 'numpy.core._multiarray_umath.compare_chararrays', 'deprecate': 'numpy.lib.utils.deprecate', 'deprecate_with_doc': 'numpy.lib.utils.deprecate_with_doc', 'disp': 'numpy.lib.function_base.disp', 'fastCopyAndTranspose': 'numpy.core._multiarray_umath.fastCopyAndTranspose', 'get_array_wrap': 'numpy.lib.shape_base.get_array_wrap', 'get_include': 'numpy.lib.utils.get_include', 'recfromcsv': 'numpy.lib.npyio.recfromcsv', 'recfromtxt': 'numpy.lib.npyio.recfromtxt', 'safe_eval': 'numpy.lib.utils.safe_eval', 'set_string_function': 'numpy.core.arrayprint.set_string_function', 'show_config': 'numpy.__config__.show', 'show_runtime': 'numpy.lib.utils.show_runtime', 'who': 'numpy.lib.utils.who'}
    allowlist = undocumented
    bad_results = check_dir(np)
    assert bad_results == allowlist