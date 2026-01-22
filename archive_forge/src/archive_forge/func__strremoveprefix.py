from __future__ import annotations
import enum
import threading
from abc import abstractmethod
from functools import wraps
from typing import (
from weakref import WeakValueDictionary
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import (
def _strremoveprefix(s: str, prefix: str) -> str:
    """str.removeprefix() is only available in Python 3.9+."""
    return s.replace(prefix, '', 1) if s.startswith(prefix) else s