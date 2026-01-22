from .. import types, utils, errors
import operator
from .templates import (AttributeTemplate, ConcreteTemplate, AbstractTemplate,
from .builtins import normalize_1d_index
@infer_global(operator.contains)
class InContainer(AbstractTemplate):
    key = operator.contains

    def generic(self, args, kws):
        cont, item = args
        if isinstance(cont, types.Container):
            return signature(types.boolean, cont, cont.dtype)