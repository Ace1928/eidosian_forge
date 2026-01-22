import warnings
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from fastapi.openapi.models import Example
from pydantic.fields import FieldInfo
from typing_extensions import Annotated, deprecated
from ._compat import PYDANTIC_V2, Undefined
class Security(Depends):

    def __init__(self, dependency: Optional[Callable[..., Any]]=None, *, scopes: Optional[Sequence[str]]=None, use_cache: bool=True):
        super().__init__(dependency=dependency, use_cache=use_cache)
        self.scopes = scopes or []