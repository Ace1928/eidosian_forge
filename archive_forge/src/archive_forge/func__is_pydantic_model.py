import warnings
from types import MappingProxyType
from typing import Any
from itemadapter._imports import attr, pydantic
def _is_pydantic_model(obj: Any) -> bool:
    if pydantic is None:
        return False
    return issubclass(obj, pydantic.BaseModel)