import ast
import collections
import contextlib
import functools
import itertools
import re
from numbers import Number
from typing import (
from more_itertools import windowed_complete
from typeguard import typechecked
from typing_extensions import Annotated, Literal
def _pl_reg_plurals(self, pair: str, stems: str, end1: str, end2: str) -> bool:
    pattern = f'({stems})({end1}\\|\\1{end2}|{end2}\\|\\1{end1})'
    return bool(re.search(pattern, pair))