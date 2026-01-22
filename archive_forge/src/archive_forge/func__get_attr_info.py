import abc
from contextlib import contextmanager
from collections import defaultdict, namedtuple
from functools import partial
from copy import copy
import warnings
from numba.core import (errors, types, typing, ir, funcdesc, rewrites,
from numba.parfors.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfors.parfor import ParforPass as _parfor_ParforPass
from numba.parfors.parfor import ParforFusionPass as _parfor_ParforFusionPass
from numba.parfors.parfor import ParforPreLoweringPass as \
from numba.parfors.parfor import Parfor
from numba.parfors.parfor_lowering import ParforLower
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (raise_on_unsupported_feature, warn_deprecated,
from numba.core import postproc
from llvmlite import binding as llvm
def _get_attr_info(self, state, expr):
    recv_type = state.typemap[expr.value.name]
    recv_type = types.unliteral(recv_type)
    matched = state.typingctx.find_matching_getattr_template(recv_type, expr.attr)
    if not matched:
        return None
    template = matched['template']
    if getattr(template, 'is_method', False):
        return None
    templates = [template]
    sig = typing.signature(matched['return_type'], recv_type)
    arg_typs = sig.args
    is_method = False
    return (templates, sig, arg_typs, is_method)