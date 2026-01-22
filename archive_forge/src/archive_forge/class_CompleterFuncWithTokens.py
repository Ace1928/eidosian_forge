from the CompletionItem.description value.
import argparse
import re
import sys
from argparse import (
from gettext import (
from typing import (
from . import (
@runtime_checkable
class CompleterFuncWithTokens(Protocol):
    """
    Function to support tab completion with the provided state of the user prompt and accepts a dictionary of prior
    arguments.
    """

    def __call__(self, text: str, line: str, begidx: int, endidx: int, *, arg_tokens: Dict[str, List[str]]={}) -> List[str]:
        ...