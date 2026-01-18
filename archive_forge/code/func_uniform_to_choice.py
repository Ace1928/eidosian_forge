from typing import Any, List, Optional, Union
import numpy as np
def uniform_to_choice(value: Any, choices: List[Any], log: bool=False, base: Optional[float]=None) -> Any:
    idx = uniform_to_integers(value, 1, len(choices), log=log, include_high=True, base=base)
    if isinstance(idx, int):
        return choices[idx - 1]
    return [choices[x - 1] for x in idx]