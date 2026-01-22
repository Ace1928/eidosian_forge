from abc import ABC
from typing import (
from typing_extensions import NotRequired
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr
class SerializedSecret(BaseSerialized):
    """Serialized secret."""
    type: Literal['secret']