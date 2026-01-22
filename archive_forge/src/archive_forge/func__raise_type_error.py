import math
import re
from typing import (
import unicodedata
from .parser import Parser
def _raise_type_error(obj):
    raise TypeError(f'{repr(obj)} is not JSON5 serializable')