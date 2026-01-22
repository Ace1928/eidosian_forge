import operator
from numba.core import types
from numba.core.typing.templates import (AbstractTemplate, AttributeTemplate,
@infer_getattr
class EnumAttribute(AttributeTemplate):
    key = types.EnumMember

    def resolve_value(self, ty):
        return ty.dtype