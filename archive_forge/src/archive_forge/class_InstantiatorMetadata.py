import collections.abc
import dataclasses
import enum
import inspect
import os
import pathlib
from collections import deque
from typing import (
from typing_extensions import Annotated, Final, Literal, get_args, get_origin
from . import _resolver
from . import _strings
from ._typing import TypeForm
from .conf import _markers
@dataclasses.dataclass
class InstantiatorMetadata:
    nargs: Optional[Union[int, Literal['*']]]
    metavar: str
    choices: Optional[Set[str]]
    action: Optional[Literal['append']]

    def check_choices(self, strings: List[str]) -> None:
        if self.choices is not None and any((s not in self.choices for s in strings)):
            raise ValueError(f'invalid choice: {strings} (choose from {self.choices}))')