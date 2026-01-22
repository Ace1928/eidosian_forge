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
def _do_work_expr(self, state, work_list, block, i, expr, inline_worker):

    def select_template(templates, args):
        if templates is None:
            return None
        impl = None
        for template in templates:
            inline_type = getattr(template, '_inline', None)
            if inline_type is None:
                continue
            if args not in template._inline_overloads:
                continue
            if not inline_type.is_never_inline:
                try:
                    impl = template._overload_func(*args)
                    if impl is None:
                        raise Exception
                    break
                except Exception:
                    continue
        else:
            return None
        return (template, inline_type, impl)
    inlinee_info = None
    if expr.op == 'getattr':
        inlinee_info = self._get_attr_info(state, expr)
    else:
        inlinee_info = self._get_callable_info(state, expr)
    if not inlinee_info:
        return False
    templates, sig, arg_typs, is_method = inlinee_info
    inlinee = select_template(templates, arg_typs)
    if inlinee is None:
        return False
    template, inlinee_type, impl = inlinee
    return self._run_inliner(state, inlinee_type, sig, template, arg_typs, expr, i, impl, block, work_list, is_method, inline_worker)