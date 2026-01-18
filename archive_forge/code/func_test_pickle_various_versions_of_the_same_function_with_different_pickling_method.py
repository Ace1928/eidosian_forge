import _collections_abc
import abc
import collections
import base64
import functools
import io
import itertools
import logging
import math
import multiprocessing
from operator import itemgetter, attrgetter
import pickletools
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import weakref
import os
import enum
import typing
from functools import wraps
import pytest
import srsly.cloudpickle as cloudpickle
from srsly.cloudpickle.compat import pickle
from srsly.cloudpickle import register_pickle_by_value
from srsly.cloudpickle import unregister_pickle_by_value
from srsly.cloudpickle import list_registry_pickle_by_value
from srsly.cloudpickle.cloudpickle import _should_pickle_by_reference
from srsly.cloudpickle.cloudpickle import _make_empty_cell, cell_set
from srsly.cloudpickle.cloudpickle import _extract_class_dict, _whichmodule
from srsly.cloudpickle.cloudpickle import _lookup_module_and_qualname
from .testutils import subprocess_pickle_echo
from .testutils import subprocess_pickle_string
from .testutils import assert_run_python_script
from .testutils import subprocess_worker
def test_pickle_various_versions_of_the_same_function_with_different_pickling_method(self):
    pytest.importorskip('_cloudpickle_testpkg')
    import _cloudpickle_testpkg
    from _cloudpickle_testpkg import package_function_with_global as f
    _original_global = _cloudpickle_testpkg.global_variable

    def _create_registry():
        _main = __import__('sys').modules['__main__']
        _main._cloudpickle_registry = {}

    def _add_to_registry(v, k):
        _main = __import__('sys').modules['__main__']
        _main._cloudpickle_registry[k] = v

    def _call_from_registry(k):
        _main = __import__('sys').modules['__main__']
        return _main._cloudpickle_registry[k]()
    try:
        with subprocess_worker(protocol=self.protocol) as w:
            w.run(_create_registry)
            w.run(_add_to_registry, f, 'f_by_ref')
            register_pickle_by_value(_cloudpickle_testpkg)
            _cloudpickle_testpkg.global_variable = 'modified global'
            w.run(_add_to_registry, f, 'f_by_val')
            assert w.run(_call_from_registry, 'f_by_ref') == _original_global
            assert w.run(_call_from_registry, 'f_by_val') == 'modified global'
    finally:
        _cloudpickle_testpkg.global_variable = _original_global
        if '_cloudpickle_testpkg' in list_registry_pickle_by_value():
            unregister_pickle_by_value(_cloudpickle_testpkg)