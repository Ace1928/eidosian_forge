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
def test_register_pickle_by_value():
    pytest.importorskip('_cloudpickle_testpkg')
    import _cloudpickle_testpkg as pkg
    import _cloudpickle_testpkg.mod as mod
    assert list_registry_pickle_by_value() == set()
    register_pickle_by_value(pkg)
    assert list_registry_pickle_by_value() == {pkg.__name__}
    register_pickle_by_value(mod)
    assert list_registry_pickle_by_value() == {pkg.__name__, mod.__name__}
    unregister_pickle_by_value(mod)
    assert list_registry_pickle_by_value() == {pkg.__name__}
    msg = f'Input should be a module object, got {pkg.__name__} instead'
    with pytest.raises(ValueError, match=msg):
        unregister_pickle_by_value(pkg.__name__)
    unregister_pickle_by_value(pkg)
    assert list_registry_pickle_by_value() == set()
    msg = f'{pkg} is not registered for pickle by value'
    with pytest.raises(ValueError, match=re.escape(msg)):
        unregister_pickle_by_value(pkg)
    msg = f'Input should be a module object, got {pkg.__name__} instead'
    with pytest.raises(ValueError, match=msg):
        register_pickle_by_value(pkg.__name__)
    dynamic_mod = types.ModuleType('dynamic_mod')
    msg = f'{dynamic_mod} was not imported correctly, have you used an `import` statement to access it?'
    with pytest.raises(ValueError, match=re.escape(msg)):
        register_pickle_by_value(dynamic_mod)