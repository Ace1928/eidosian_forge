import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
@property
def title_pattern(self) -> str:
    return f'^\\.\\.\\s*({self.title})\\s*::'