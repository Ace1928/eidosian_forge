import sys
import weakref
from collections import defaultdict
from contextlib import contextmanager
from functools import partial
from itertools import groupby
from numbers import Number
from types import FunctionType
import numpy as np
import pandas as pd
import param
from packaging.version import Version
from .core import util
from .core.ndmapping import UniformNdMapping
@classmethod
def _process_streams(cls, streams):
    """
        Processes a list of streams promoting Parameterized objects and
        methods to Param based streams.
        """
    parameterizeds = defaultdict(set)
    valid, invalid = ([], [])
    for s in streams:
        if isinstance(s, partial):
            s = s.func
        if isinstance(s, Stream):
            pass
        elif isinstance(s, param.Parameter):
            s = Params(s.owner, [s.name])
        elif isinstance(s, param.Parameterized):
            s = Params(s)
        elif util.is_param_method(s):
            if not hasattr(s, '_dinfo'):
                continue
            s = ParamMethod(s)
        elif isinstance(s, FunctionType) and hasattr(s, '_dinfo'):
            deps = s._dinfo
            dep_params = list(deps['dependencies']) + list(deps.get('kw', {}).values())
            rename = {(p.owner, p.name): k for k, p in deps.get('kw', {}).items()}
            s = Params(parameters=dep_params, rename=rename)
        else:
            if util.param_version > util.Version('2.0.0rc1'):
                deps = param.parameterized.resolve_ref(s)
            else:
                deps = None
            if deps:
                s = Params(parameters=deps)
            else:
                invalid.append(s)
                continue
        if isinstance(s, Params):
            pid = id(s.parameterized)
            overlap = set(s.parameters) & parameterizeds[pid]
            if overlap:
                pname = type(s.parameterized).__name__
                param.main.param.warning('The {} parameter(s) on the {} object have already been supplied in another stream. Ensure that the supplied streams only specify each parameter once, otherwise multiple events will be triggered when the parameter changes.'.format(sorted([p.name for p in overlap]), pname))
            parameterizeds[pid] |= set(s.parameters)
        valid.append(s)
    return (valid, invalid)