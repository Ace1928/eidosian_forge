from __future__ import annotations
import builtins as builtin_mod
import enum
import glob
import inspect
import itertools
import keyword
import os
import re
import string
import sys
import tokenize
import time
import unicodedata
import uuid
import warnings
from ast import literal_eval
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property, partial
from types import SimpleNamespace
from typing import (
from IPython.core.guarded_eval import guarded_eval, EvaluationContext
from IPython.core.error import TryNext
from IPython.core.inputtransformer2 import ESC_MAGIC
from IPython.core.latex_symbols import latex_symbols, reverse_latex_symbol
from IPython.core.oinspect import InspectColors
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils import generics
from IPython.utils.decorators import sphinx_options
from IPython.utils.dir2 import dir2, get_real_method
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.path import ensure_dir_exists
from IPython.utils.process import arg_split
from traitlets import (
from traitlets.config.configurable import Configurable
import __main__
class _DictKeyState(enum.Flag):
    """Represent state of the key match in context of other possible matches.

    - given `d1 = {'a': 1}` completion on `d1['<tab>` will yield `{'a': END_OF_ITEM}` as there is no tuple.
    - given `d2 = {('a', 'b'): 1}`: `d2['a', '<tab>` will yield `{'b': END_OF_TUPLE}` as there is no tuple members to add beyond `'b'`.
    - given `d3 = {('a', 'b'): 1}`: `d3['<tab>` will yield `{'a': IN_TUPLE}` as `'a'` can be added.
    - given `d4 = {'a': 1, ('a', 'b'): 2}`: `d4['<tab>` will yield `{'a': END_OF_ITEM & END_OF_TUPLE}`
    """
    BASELINE = 0
    END_OF_ITEM = enum.auto()
    END_OF_TUPLE = enum.auto()
    IN_TUPLE = enum.auto()