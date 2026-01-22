import inspect
import itertools
import re
import typing as T
from textwrap import dedent
from .common import (
class YieldsSection(ReturnsSection):
    """Parser for numpydoc generator "yields" sections."""
    is_generator = True