from pythran.conversion import to_ast
from pythran.interval import UNKNOWN_RANGE, bool_values
from pythran.types.signature import extract_combiner
from pythran.typing import Any, Union, Fun, Generator
import gast as ast
class Intrinsic(object):
    """
    Model any Method/Function.

    Its member variables are:

    - argument_effects that describes the effect of the function on its
      argument (either UpdateEffect, ReadEffect or ReadOnceEffect)
    - global_effects that describes whether the function has side effects
    - return_alias that describes the aliasing between the return value
      and the parameters. The lambda returns an ast expression, generally
      depending on the node arguments (see dict.setdefault)
    - args that describes the name and default value of each arg, using the
      same representation as ast.FunctionDef, i.e. ast.arguments
    """

    def __init__(self, **kwargs):
        self.argument_effects = kwargs.get('argument_effects', (UpdateEffect(),) * DefaultArgNum)
        self.global_effects = kwargs.get('global_effects', False)
        self.return_alias = kwargs.get('return_alias', lambda x: {UnboundValue})
        self.args = ast.arguments([ast.Name(n, ast.Param(), None, None) for n in kwargs.get('args', [])], [], None, [ast.Name(n, ast.Param(), None, None) for n in kwargs.get('kwonlyargs', [])], [], None, [to_ast(d) for d in kwargs.get('defaults', [])])
        self.return_range = kwargs.get('return_range', lambda call: UNKNOWN_RANGE)
        self.return_range_content = kwargs.get('return_range_content', lambda c: UNKNOWN_RANGE)

    def isliteral(self):
        return False

    def isstatic(self):
        """static <=> value is known at compile time"""
        return False

    def isfunction(self):
        return False

    def isstaticfunction(self):
        return False

    def ismethod(self):
        return False

    def isattribute(self):
        return False

    def isconst(self):
        return not any((isinstance(x, UpdateEffect) for x in self.argument_effects)) and (not self.global_effects)

    def isreadonce(self, n):
        return isinstance(self.argument_effects[n], ReadOnceEffect)

    def combiner(self, s, node):
        pass