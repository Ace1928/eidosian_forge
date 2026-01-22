import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def _standardize_iterables(node):
    """Converts the given iterables to a {field: function} dictionary,
    if necessary, where the function returns a list."""
    if not node.iterables:
        return
    iterables = node.iterables
    fields = set(node.inputs.copyable_trait_names())
    if node.synchronize:
        if len(iterables) == 2:
            first, last = iterables
            if all((isinstance(item, (str, bytes)) and item in fields for item in first)):
                iterables = _transpose_iterables(first, last)
    if isinstance(iterables, tuple):
        iterables = [iterables]
    _validate_iterables(node, iterables, fields)
    if isinstance(iterables, list):
        if not node.itersource:

            def make_field_func(*pair):
                return (pair[0], lambda: pair[1])
            iter_items = [make_field_func(*field_value1) for field_value1 in iterables]
            iterables = dict(iter_items)
    node.iterables = iterables