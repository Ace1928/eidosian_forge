from pythran.conversion import to_ast
from pythran.interval import UNKNOWN_RANGE, bool_values
from pythran.types.signature import extract_combiner
from pythran.typing import Any, Union, Fun, Generator
import gast as ast
class ReadOnceMethodIntr(ConstMethodIntr):

    def __init__(self, **kwargs):
        super(ReadOnceMethodIntr, self).__init__(argument_effects=(ReadOnceEffect(),) * DefaultArgNum, **kwargs)