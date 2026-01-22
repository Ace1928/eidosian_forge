import sys
from math import trunc
from typing import (
def _name_to_ordinal(self, lst: Sequence[str]) -> Dict[str, int]:
    return {elem.lower(): i for i, elem in enumerate(lst[1:], 1)}