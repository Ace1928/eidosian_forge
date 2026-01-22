from abc import ABC
from typing import (
from typing_extensions import NotRequired
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr
class SerializedNotImplemented(BaseSerialized):
    """Serialized not implemented."""
    type: Literal['not_implemented']
    repr: Optional[str]