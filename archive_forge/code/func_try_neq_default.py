from abc import ABC
from typing import (
from typing_extensions import NotRequired
from langchain_core.pydantic_v1 import BaseModel, PrivateAttr
def try_neq_default(value: Any, key: str, model: BaseModel) -> bool:
    """Try to determine if a value is different from the default.

    Args:
        value: The value.
        key: The key.
        model: The model.

    Returns:
        Whether the value is different from the default.
    """
    try:
        return model.__fields__[key].get_default() != value
    except Exception:
        return True