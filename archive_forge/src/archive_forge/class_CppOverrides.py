import contextlib
import dataclasses
import functools
import itertools
import logging
import math
import re
import sys
from copy import copy, deepcopy
from typing import Dict, List, Optional, Set, Tuple, Union
import sympy
import torch
import torch.fx
from torch._inductor import dependencies
from torch._inductor.ir import StorageBox, TensorBox
from torch._prims_common import is_float_dtype
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.value_ranges import bound_sympy, ValueRanges
from .. import codecache, config, ir, metrics
from ..codegen.wrapper import WrapperCodeGen
from ..optimize_indexing import range_expressable_in_32_bits
from ..scheduler import BaseScheduling, SchedulerNode
from ..utils import (
from ..virtualized import ops, V
from .common import (
class CppOverrides(OpOverrides):
    """Map element-wise ops to C++"""

    @staticmethod
    def add(a, b):
        return f'decltype({a})({a} + {b})'

    @staticmethod
    def sub(a, b):
        return f'decltype({a})({a} - {b})'

    @staticmethod
    def mul(a, b):
        return f'decltype({a})({a} * {b})'

    @staticmethod
    def to_dtype(x, dtype, src_dtype=None):
        assert dtype in DTYPE_TO_CPP, f'{dtype} missing from {__name__}.DTYPE_TO_CPP'
        return f'c10::convert<{DTYPE_TO_CPP[dtype]}>({x})'

    @staticmethod
    def to_dtype_bitcast(x, dtype):
        assert dtype in DTYPE_TO_CPP, f'{dtype} missing from {__name__}.DTYPE_TO_CPP'
        return f'c10::bit_cast<{DTYPE_TO_CPP[dtype]}>({x})'

    @staticmethod
    def abs(x):
        return f'std::abs({x})'

    @staticmethod
    def sin(x):
        return f'std::sin({x})'

    @staticmethod
    def cos(x):
        return f'std::cos({x})'

    @staticmethod
    def neg(x):
        return f'decltype({x})(-{x})'

    @staticmethod
    def exp(x):
        return f'std::exp({x})'

    @staticmethod
    def exp2(x):
        return f'std::exp2({x})'

    @staticmethod
    def expm1(x):
        return f'std::expm1({x})'

    @staticmethod
    def erf(x):
        return f'std::erf({x})'

    @staticmethod
    def erfc(x):
        return f'std::erfc({x})'

    @staticmethod
    def erfinv(x):
        return f'calc_erfinv({x})'

    @staticmethod
    def sqrt(x):
        return f'std::sqrt({x})'

    @staticmethod
    def rsqrt(x):
        return f'1 / std::sqrt({x})'

    @staticmethod
    def log1p(x):
        bug = config.cpp.inject_log1p_bug_TESTING_ONLY
        if bug == 'accuracy':
            return f'{x} + decltype({x})(1)'
        elif bug is None:
            return f'std::log1p({x})'
        else:
            raise AssertionError(f'unrecognized config cpp.inject_log1p_bug_TESTING_ONLY = {bug!r}')

    @staticmethod
    def tan(x):
        return f'std::tan({x})'

    @staticmethod
    def tanh(x):
        return f'std::tanh({x})'

    @staticmethod
    def signbit(x):
        return f'std::signbit({x})'

    @staticmethod
    def pow(a, b):
        return f'std::pow({a}, {b})'

    @staticmethod
    def log(x):
        return f'std::log({x})'

    @staticmethod
    def round(x):
        return f'std::nearbyint({x})'

    @staticmethod
    def floor(x):
        return f'std::floor({x})'

    @staticmethod
    def floordiv(a, b):
        quot = f'{a} / {b}'
        rem = f'{a} % {b}'
        return f'(({a} < 0) != ({b} < 0) ? ({rem} != 0 ? {quot} - 1 : {quot}) : {quot})'

    @staticmethod
    def ceil(x):
        return f'std::ceil({x})'

    @staticmethod
    def trunc(x):
        return f'std::trunc({x})'

    @staticmethod
    def truncdiv(a, b):
        return f'{a} / {b}'

    @staticmethod
    def fmod(a, b):
        return f'std::fmod({a}, {b})'

    @staticmethod
    def isinf(x):
        return f'std::isinf({x})'

    @staticmethod
    def isnan(x):
        return f'std::isnan({x})'

    @staticmethod
    def lgamma(x):
        return f'std::lgamma({x})'

    @staticmethod
    def acos(x):
        return f'std::acos({x})'

    @staticmethod
    def acosh(x):
        return f'std::acosh({x})'

    @staticmethod
    def cosh(x):
        return f'std::cosh({x})'

    @staticmethod
    def sinh(x):
        return f'std::sinh({x})'

    @staticmethod
    def asin(x):
        return f'std::asin({x})'

    @staticmethod
    def asinh(x):
        return f'std::asinh({x})'

    @staticmethod
    def atan2(x, y):
        return f'std::atan2({x}, {y})'

    @staticmethod
    def atan(x):
        return f'std::atan({x})'

    @staticmethod
    def atanh(x):
        return f'std::atanh({x})'

    @staticmethod
    def copysign(x, y):
        return f'std::copysign({x}, {y})'

    @staticmethod
    def hypot(x, y):
        return f'std::hypot({x}, {y})'

    @staticmethod
    def log10(x):
        return f'std::log10({x})'

    @staticmethod
    def nextafter(x, y):
        return f'std::nextafter({x}, {y})'

    @staticmethod
    def relu(x):
        bug = config.cpp.inject_relu_bug_TESTING_ONLY
        if bug == 'compile_error':
            return 'compile error!'
        elif bug == 'runtime_error':
            return f'{x}; throw 1'
        elif bug == 'accuracy':
            return f'{x} + decltype({x})(1)'
        elif bug is None:
            return f'{x} * ({x}>0)'
        else:
            raise AssertionError(f'unrecognized config cpp.inject_relu_bug_TESTING_ONLY = {bug!r}')

    @staticmethod
    def minimum(a, b):
        return f'min_propagate_nan({a}, {b})'

    @staticmethod
    def maximum(a, b):
        return f'max_propagate_nan({a}, {b})'

    @staticmethod
    def where(a, b, c):
        return f'{a} ? {b} : {c}'

    @staticmethod
    def mod(a, b):
        return f'mod({a}, {b})'

    @staticmethod
    def constant(val, dtype):
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx and opt_ctx.dtype is not None
        dtype = opt_ctx.dtype
        if dtype in DTYPE_LOWP_FP:
            dtype = torch.float32
        return value_to_cpp(val, DTYPE_TO_CPP[dtype])

    @staticmethod
    def index_expr(expr, dtype):
        opt_ctx: OptimizationContext = get_current_node_opt_ctx()
        assert opt_ctx and opt_ctx.dtype is not None
        dtype = opt_ctx.dtype
        return ops.to_dtype(cexpr(V.kernel.rename_indexing(expr)), dtype)

    @staticmethod
    def masked(mask, body, other):
        code = BracesBuffer()
        body_var = V.kernel.cse.newvar()
        code.writeline(f'auto {body_var} = [&]')
        with V.kernel.swap_buffers(code), code.indent():
            result = body()
            code.writeline(f'return {result};')
        code.writeline(';')
        V.kernel.compute.splice(code)
        other_code = value_to_cpp(other, f'decltype({body_var}())')
        return f'{mask} ? {body_var}() : {other_code}'

    @staticmethod
    def logical_and(a, b):
        return f'{a} && {b}'

    @staticmethod
    def logical_not(a):
        return f'!{a}'

    @staticmethod
    def logical_or(a, b):
        return f'{a} || {b}'

    @staticmethod
    def logical_xor(a, b):
        return f'{a} != {b}'

    @staticmethod
    def bitwise_and(a, b):
        return f'decltype({a})({a} & {b})'

    @staticmethod
    def bitwise_not(a):
        return f'decltype({a})(~{a})'

    @staticmethod
    def bitwise_or(a, b):
        return f'decltype({a})({a} | {b})'

    @staticmethod
    def bitwise_xor(a, b):
        return f'decltype({a})({a} ^ {b})'

    @staticmethod
    def bitwise_left_shift(a, b):
        return f'decltype({a})({a} << {b})'

    @staticmethod
    def bitwise_right_shift(a, b):
        return f'decltype({a})({a} >> {b})'

    @staticmethod
    def rand(seed: sympy.Expr, offset: sympy.Expr):
        return f'normalized_rand_cpu({seed}, {offset})'

    @staticmethod
    def randn(seed: sympy.Expr, offset: sympy.Expr):
        return f'randn_cpu({seed}, {offset})'

    @staticmethod
    def randint64(seed: sympy.Expr, offset: sympy.Expr, low, high):
        return f'randint64_cpu({seed}, {offset}, {low}, {high})'

    @staticmethod
    def sigmoid(x):
        return f'decltype({x})(1) / (decltype({x})(1) + std::exp(-{x}))'

    @staticmethod
    def sign(x):
        code = BracesBuffer()
        left = V.kernel.cse.newvar()
        right = V.kernel.cse.newvar()
        result = V.kernel.cse.newvar()
        scalar_zero = f'decltype({x})(0)'
        scalar_one = f'decltype({x})(1)'
        code.writeline(f'auto {left} = {x} > 0 ? {scalar_one} : {scalar_zero};')
        code.writeline(f'auto {right} = {x} < 0 ? {scalar_one} : {scalar_zero};')
        code.writeline(f'auto {result} = {left} - {right};')
        V.kernel.compute.splice(code)
        return result