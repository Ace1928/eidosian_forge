import sys
from math import trunc
from typing import (
def _format_relative(self, humanized: str, timeframe: TimeFrameLiteral, delta: Union[float, int]) -> str:
    """Lao normally doesn't have any space between words"""
    if timeframe == 'now':
        return humanized
    direction = self.past if delta < 0 else self.future
    relative_string = direction.format(humanized)
    if timeframe == 'seconds':
        relative_string = relative_string.replace(' ', '')
    return relative_string