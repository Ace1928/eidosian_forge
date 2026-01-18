from typing import Any, List, Optional, Union
import numpy as np
def uniform_to_discrete(value: Any, low: float, high: float, q: float, log: bool=False, include_high: bool=True, base: Optional[float]=None) -> Any:
    if low >= high:
        return low if np.isscalar(value) else np.full(np.shape(value))
    _high = adjust_high(low, high, q, include_high=include_high)
    _value = uniform_to_continuous(value, low, _high, log=log, base=base)
    return np.floor((_value - low) / q) * q + low