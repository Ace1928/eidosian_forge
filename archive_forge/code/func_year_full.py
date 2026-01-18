import sys
from math import trunc
from typing import (
def year_full(self, year: int) -> str:
    """Lao always use Buddhist Era (BE) which is CE + 543"""
    year += self.BE_OFFSET
    return f'{year:04d}'