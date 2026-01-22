import re
import sys
from ast import literal_eval
from functools import total_ordering
from typing import NamedTuple, Sequence, Union
class _PythonVersionInfo(NamedTuple):
    major: int
    minor: int