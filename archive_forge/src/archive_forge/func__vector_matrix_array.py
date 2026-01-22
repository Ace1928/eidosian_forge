import array
import contextlib
import os
import types
import typing
import rpy2.rinterface as rinterface
import rpy2.rinterface_lib.embedded
import rpy2.rinterface_lib.openrlib
import rpy2.rlike.container as rlc
from rpy2.robjects.robject import RObjectMixin, RObject
import rpy2.robjects.functions
from rpy2.robjects.environments import (Environment,
from rpy2.robjects.methods import methods_env
from rpy2.robjects.methods import RS4
from . import conversion
from . import vectors
from . import language
from rpy2.rinterface import (Sexp,
from rpy2.robjects.functions import Function
from rpy2.robjects.functions import SignatureTranslatedFunction
def _vector_matrix_array(obj, vector_cls: typing.Type[VT], matrix_cls: typing.Type[MT], array_cls: typing.Type[AT]) -> typing.Union[typing.Type[VT], typing.Type[MT], typing.Type[AT]]:
    try:
        dim = obj.do_slot('dim')
        if len(dim) == 2:
            return matrix_cls
        else:
            return array_cls
    except Exception:
        return vector_cls