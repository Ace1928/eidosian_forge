import collections
from itertools import repeat
from typing import List, Dict, Any
def _reverse_repeat_tuple(t, n):
    """Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple((x for x in reversed(t) for _ in range(n)))