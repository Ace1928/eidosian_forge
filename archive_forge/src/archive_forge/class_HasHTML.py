import functools
import re
import string
import sys
import typing as t
class HasHTML(te.Protocol):

    def __html__(self) -> str:
        pass