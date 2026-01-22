import dataclasses
import sys
import types
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Protocol, Union
from typing_extensions import TypeAlias
import torch
class ProfilerStartHook(Protocol):

    def __call__(self, name: str) -> Any:
        ...