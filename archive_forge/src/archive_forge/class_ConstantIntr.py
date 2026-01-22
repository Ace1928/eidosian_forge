from pythran.conversion import to_ast
from pythran.interval import UNKNOWN_RANGE, bool_values
from pythran.types.signature import extract_combiner
from pythran.typing import Any, Union, Fun, Generator
import gast as ast
class ConstantIntr(Intrinsic):
    """
    Internal representation for any constant.

    Examples
    --------
    >> math.pi
    """

    def __init__(self, **kwargs):
        """ Forward arguments and remove arguments effects. """
        kwargs['argument_effects'] = ()
        super(ConstantIntr, self).__init__(**kwargs)

    def isliteral(self):
        """ Mark this intrinsic as a literal. """
        return True