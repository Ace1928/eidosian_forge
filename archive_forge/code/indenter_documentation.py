from abc import ABC, abstractmethod
from typing import List, Iterator
from .exceptions import LarkError
from .lark import PostLex
from .lexer import Token
Provides a post-lexer for implementing Python-style indentation.