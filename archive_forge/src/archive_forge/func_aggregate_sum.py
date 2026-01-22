import sys
from typing import Union
def aggregate_sum(samples: Sequence[Number], precision: int=2) -> Union[float, int]:
    if isinstance(samples[-1], int):
        return sum(samples)
    return round(sum(samples), precision)