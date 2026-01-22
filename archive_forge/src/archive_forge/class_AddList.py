import operator
from numba.core import types
from .templates import (ConcreteTemplate, AbstractTemplate, AttributeTemplate,
from numba.core.typing import collections
@infer_global(operator.add)
class AddList(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) == 2:
            a, b = args
            if isinstance(a, types.List) and isinstance(b, types.List):
                unified = self.context.unify_pairs(a, b)
                if unified is not None:
                    return signature(unified, a, b)