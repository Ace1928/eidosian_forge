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
@pytest.mark.skip(reason='Requires pytest -s to pass')
def test_pickle_constructs_from_module_registered_for_pickling_by_value(self):
    _prev_sys_path = sys.path.copy()
    try:
        _mock_interactive_session_cwd = os.path.dirname(__file__)
        _maybe_remove(sys.path, '')
        _maybe_remove(sys.path, _mock_interactive_session_cwd)
        sys.path.insert(0, _mock_interactive_session_cwd)
        with subprocess_worker(protocol=self.protocol) as w:
            w.run(lambda p: sys.path.remove(p), _mock_interactive_session_cwd)
            import mock_local_folder.mod as mod
            from mock_local_folder.mod import local_function, LocalT, LocalClass
            with pytest.raises(ImportError):
                w.run(lambda: __import__('mock_local_folder.mod'))
            with pytest.raises(ImportError):
                w.run(lambda: __import__('mock_local_folder.subfolder.mod'))
            for o in [mod, local_function, LocalT, LocalClass]:
                with pytest.raises(ImportError):
                    w.run(lambda: o)
            register_pickle_by_value(mod)
            assert w.run(lambda: local_function()) == local_function()
            assert w.run(lambda: LocalT.__name__) == LocalT.__name__
            assert w.run(lambda: LocalClass().method()) == LocalClass().method()
            assert w.run(lambda: mod.local_function()) == local_function()
            from mock_local_folder.subfolder.submod import LocalSubmodClass, LocalSubmodT, local_submod_function
            _t, _func, _class = (LocalSubmodT, local_submod_function, LocalSubmodClass)
            with pytest.raises(ImportError):
                w.run(lambda: __import__('mock_local_folder.subfolder.mod'))
            with pytest.raises(ImportError):
                w.run(lambda: local_submod_function)
            unregister_pickle_by_value(mod)
            with pytest.raises(ImportError):
                w.run(lambda: local_function)
            with pytest.raises(ImportError):
                w.run(lambda: __import__('mock_local_folder.mod'))
            import mock_local_folder
            register_pickle_by_value(mock_local_folder)
            assert w.run(lambda: local_function()) == local_function()
            assert w.run(lambda: _func()) == _func()
            unregister_pickle_by_value(mock_local_folder)
            with pytest.raises(ImportError):
                w.run(lambda: local_function)
            with pytest.raises(ImportError):
                w.run(lambda: local_submod_function)
            import mock_local_folder.subfolder.submod
            register_pickle_by_value(mock_local_folder.subfolder.submod)
            assert w.run(lambda: _func()) == _func()
            assert w.run(lambda: _t.__name__) == _t.__name__
            assert w.run(lambda: _class().method()) == _class().method()
            with pytest.raises(ImportError):
                w.run(lambda: local_function)
            with pytest.raises(ImportError):
                w.run(lambda: __import__('mock_local_folder.mod'))
            unregister_pickle_by_value(mock_local_folder.subfolder.submod)
            with pytest.raises(ImportError):
                w.run(lambda: local_submod_function)
            import mock_local_folder.subfolder
            register_pickle_by_value(mock_local_folder.subfolder)
            assert w.run(lambda: _func()) == _func()
            assert w.run(lambda: _t.__name__) == _t.__name__
            assert w.run(lambda: _class().method()) == _class().method()
            unregister_pickle_by_value(mock_local_folder.subfolder)
    finally:
        _fname = 'mock_local_folder'
        sys.path = _prev_sys_path
        for m in [_fname, f'{_fname}.mod', f'{_fname}.subfolder', f'{_fname}.subfolder.submod']:
            mod = sys.modules.pop(m, None)
            if mod and mod.__name__ in list_registry_pickle_by_value():
                unregister_pickle_by_value(mod)