import py
import sys, inspect
from compiler import parse, ast, pycodegen
from py._code.assertion import BuiltinAssertionError, _format_explanation
import types
class BinaryArith(Interpretable):
    __view__ = astclass

    def eval(self, frame, astpattern=astpattern):
        left = Interpretable(self.left)
        left.eval(frame)
        right = Interpretable(self.right)
        right.eval(frame)
        self.explanation = astpattern.replace('__exprinfo_left', left.explanation).replace('__exprinfo_right', right.explanation)
        try:
            self.result = frame.eval(astpattern, __exprinfo_left=left.result, __exprinfo_right=right.result)
        except passthroughex:
            raise
        except:
            raise Failure(self)