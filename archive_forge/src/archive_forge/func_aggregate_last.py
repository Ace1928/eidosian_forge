import sys
from typing import Union
def aggregate_last(samples: Sequence[Number], precision: int=2) -> Union[float, int]:
    if isinstance(samples[-1], int):
        return samples[-1]
    return round(samples[-1], precision)