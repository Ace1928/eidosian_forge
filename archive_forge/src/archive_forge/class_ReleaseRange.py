import io
import logging
import os
import pkgutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import Logger
from typing import IO, Any, Iterable, Iterator, List, Optional, Tuple, Union, cast
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.tokenize import GoodTokenInfo
from blib2to3.pytree import NL
from . import grammar, parse, pgen, token, tokenize
@dataclass
class ReleaseRange:
    start: int
    end: Optional[int] = None
    tokens: List[Any] = field(default_factory=list)

    def lock(self) -> None:
        total_eaten = len(self.tokens)
        self.end = self.start + total_eaten