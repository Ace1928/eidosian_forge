import itertools
import numpy as np
import operator
from numba.core import types, errors
from numba import prange
from numba.parfors.parfor import internal_prange
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cpython.builtins import get_type_min_value, get_type_max_value
from numba.core.extending import (
@register_model(IndexValueType)
class IndexValueModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        members = [('index', types.intp), ('value', fe_type.val_typ)]
        models.StructModel.__init__(self, dmm, fe_type, members)