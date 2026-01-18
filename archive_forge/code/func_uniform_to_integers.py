from typing import Any, List, Optional, Union
import numpy as np
def uniform_to_integers(value: Any, low: int, high: int, q: int=1, log: bool=False, include_high: bool=True, base: Optional[float]=None) -> Union[int, List[int]]:
    res = np.round(uniform_to_discrete(value, low, high, q=q, log=log, include_high=include_high, base=base))
    if np.isscalar(res):
        return int(res)
    return [int(x) for x in res]