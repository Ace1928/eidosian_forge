import numpy as np
from numba.core import types, cgutils
from numba.core.datamodel import models
from numba.core.extending import (
from numba.core.imputils import impl_ret_borrowed
@register_model(IndexType)
class IndexModel(models.StructModel):

    def __init__(self, dmm, fe_type):
        members = [('data', fe_type.as_array)]
        models.StructModel.__init__(self, dmm, fe_type, members)