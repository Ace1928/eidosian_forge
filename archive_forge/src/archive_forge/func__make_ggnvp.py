from __future__ import absolute_import
from functools import partial
from collections import OrderedDict
import warnings
from .wrap_util import unary_to_nary
from .builtins import tuple as atuple
from .core import make_vjp as _make_vjp, make_jvp as _make_jvp
from .extend import primitive, defvjp_argnum, vspace
import autograd.numpy as np
@unary_to_nary
def _make_ggnvp(f, x):
    f_vjp, f_x = _make_vjp(f, x)
    g_hvp, grad_g_x = _make_vjp(grad(g), f_x)
    f_jvp, _ = _make_vjp(f_vjp, vspace(grad_g_x).zeros())

    def ggnvp(v):
        return f_vjp(g_hvp(f_jvp(v)))
    return ggnvp