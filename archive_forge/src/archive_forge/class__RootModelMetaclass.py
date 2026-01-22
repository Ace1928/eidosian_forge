from __future__ import annotations as _annotations
import typing
from copy import copy, deepcopy
from pydantic_core import PydanticUndefined
from . import PydanticUserError
from ._internal import _model_construction, _repr
from .main import BaseModel, _object_setattr
@dataclass_transform(kw_only_default=False, field_specifiers=(PydanticModelField,))
class _RootModelMetaclass(_model_construction.ModelMetaclass):
    ...