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
def test_unrelated_faulty_module(self):
    for base_class in (object, types.ModuleType):
        for module_name in ['_missing_module', None]:

            class FaultyModule(base_class):

                def __getattr__(self, name):
                    raise Exception()

            class Foo:
                __module__ = module_name

                def foo(self):
                    return 'it works!'

            def foo():
                return 'it works!'
            foo.__module__ = module_name
            if base_class is types.ModuleType:
                faulty_module = FaultyModule('_faulty_module')
            else:
                faulty_module = FaultyModule()
            sys.modules['_faulty_module'] = faulty_module
            try:
                self.assertEqual(pickle_depickle(Foo()).foo(), 'it works!')
                cloned = pickle_depickle(foo, protocol=self.protocol)
                self.assertEqual(cloned(), 'it works!')
            finally:
                sys.modules.pop('_faulty_module', None)