import ast
import builtins
import dis
import enum
import inspect
import re
import typing
import warnings
from textwrap import dedent
from typing import Type
import torch
from torch._C import (
from torch._sources import get_source_lines_and_file
from .._jit_internal import (  # type: ignore[attr-defined]
from ._state import _get_script_class
from torch._ops import OpOverloadPacket
def try_real_annotations(fn, loc):
    """Try to use the Py3.5+ annotation syntax to get the type."""
    try:
        sig = inspect.signature(fn)
    except ValueError:
        return None
    all_annots = [sig.return_annotation] + [p.annotation for p in sig.parameters.values()]
    if all((ann is sig.empty for ann in all_annots)):
        return None
    arg_types = [ann_to_type(p.annotation, loc) for p in sig.parameters.values()]
    return_type = ann_to_type(sig.return_annotation, loc)
    return (arg_types, return_type)