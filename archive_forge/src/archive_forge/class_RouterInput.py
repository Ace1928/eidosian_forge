from __future__ import annotations
from typing import (
from typing_extensions import TypedDict
from langchain_core.runnables.base import (
from langchain_core.runnables.config import (
from langchain_core.runnables.utils import (
class RouterInput(TypedDict):
    """Router input.

    Attributes:
        key: The key to route on.
        input: The input to pass to the selected runnable.
    """
    key: str
    input: Any