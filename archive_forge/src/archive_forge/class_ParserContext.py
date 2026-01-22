from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class ParserContext:

    def __init__(self, ctx=None):
        self.ctx = _get_ctx(ctx)
        self.pctx = Z3_mk_parser_context(self.ctx.ref())
        Z3_parser_context_inc_ref(self.ctx.ref(), self.pctx)

    def __del__(self):
        if self.ctx.ref() is not None and self.pctx is not None and (Z3_parser_context_dec_ref is not None):
            Z3_parser_context_dec_ref(self.ctx.ref(), self.pctx)
            self.pctx = None

    def add_sort(self, sort):
        Z3_parser_context_add_sort(self.ctx.ref(), self.pctx, sort.as_ast())

    def add_decl(self, decl):
        Z3_parser_context_add_decl(self.ctx.ref(), self.pctx, decl.as_ast())

    def from_string(self, s):
        return AstVector(Z3_parser_context_from_string(self.ctx.ref(), self.pctx, s), self.ctx)