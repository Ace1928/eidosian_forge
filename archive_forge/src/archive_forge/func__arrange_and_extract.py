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
def _arrange_and_extract(self, results: Dict[str, MatcherResult], skip_matchers: Set[str], abort_if_offset_changes: bool):
    sortable: List[AnyMatcherCompletion] = []
    ordered: List[AnyMatcherCompletion] = []
    most_recent_fragment = None
    for identifier, result in results.items():
        if identifier in skip_matchers:
            continue
        if not result['completions']:
            continue
        if not most_recent_fragment:
            most_recent_fragment = result['matched_fragment']
        if abort_if_offset_changes and result['matched_fragment'] != most_recent_fragment:
            break
        if result.get('ordered', False):
            ordered.extend(result['completions'])
        else:
            sortable.extend(result['completions'])
    if not most_recent_fragment:
        most_recent_fragment = ''
    return (most_recent_fragment, [m.text for m in self._deduplicate(ordered + self._sort(sortable))])