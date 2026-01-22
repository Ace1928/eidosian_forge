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
def _deduplicate_completions(text: str, completions: _IC) -> _IC:
    """
    Deduplicate a set of completions.

    .. warning::

        Unstable

        This function is unstable, API may change without warning.

    Parameters
    ----------
    text : str
        text that should be completed.
    completions : Iterator[Completion]
        iterator over the completions to deduplicate

    Yields
    ------
    `Completions` objects
    Completions coming from multiple sources, may be different but end up having
    the same effect when applied to ``text``. If this is the case, this will
    consider completions as equal and only emit the first encountered.
    Not folded in `completions()` yet for debugging purpose, and to detect when
    the IPython completer does return things that Jedi does not, but should be
    at some point.
    """
    completions = list(completions)
    if not completions:
        return
    new_start = min((c.start for c in completions))
    new_end = max((c.end for c in completions))
    seen = set()
    for c in completions:
        new_text = text[new_start:c.start] + c.text + text[c.end:new_end]
        if new_text not in seen:
            yield c
            seen.add(new_text)