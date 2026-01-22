from abc import ABC, abstractmethod
from typing import List, Iterator
from .exceptions import LarkError
from .lark import PostLex
from .lexer import Token
@property
@abstractmethod
def OPEN_PAREN_types(self) -> List[str]:
    raise NotImplementedError()