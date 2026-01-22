from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
class SpecialPythonTypeConstructor(PyObjectType, PythonTypeConstructorMixin):
    """
    For things like ClassVar, Optional, etc, which are not types and disappear during type analysis.
    """

    def __init__(self, name):
        super(SpecialPythonTypeConstructor, self).__init__()
        self.set_python_type_constructor_name(name)
        self.modifier_name = name

    def __repr__(self):
        return self.name

    def resolve(self):
        return self

    def specialize_here(self, pos, env, template_values=None):
        if len(template_values) != 1:
            error(pos, "'%s' takes exactly one template argument." % self.name)
            return error_type
        if template_values[0] is None:
            return None
        return template_values[0].resolve()