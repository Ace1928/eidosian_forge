from pythran.analyses import ConstantExpressions, ASTMatcher
from pythran.passmanager import Transformation
from pythran.tables import MODULES
from pythran.conversion import to_ast, ConversionError, ToNotEval, mangle
from pythran.analyses.ast_matcher import DamnTooLongPattern
from pythran.syntax import PythranSyntaxError
from pythran.utils import isintegral, isnum
from pythran.config import cfg
import builtins
import gast as ast
from copy import deepcopy
import logging
import sys
class FunctionDefWrapper(object):

    def __init__(self, evaluator, func):
        self.evaluator = evaluator
        self.func = func

    def __call__(self, *args):
        missing_args = len(args) - len(self.func.args.args)
        if missing_args:
            defaults = tuple((self.evaluator.visit(default) for default in self.func.args.defaults[missing_args:]))
        else:
            defaults = ()
        locals = {arg.id: argv for arg, argv in zip(self.func.args.args, args + defaults)}
        curr_locals, self.evaluator.locals = (self.evaluator.locals, locals)
        try:
            for stmt in self.func.body:
                self.evaluator.visit(stmt)
            res = locals.get('@', None)
            return res
        finally:
            self.evaluator.locals = curr_locals