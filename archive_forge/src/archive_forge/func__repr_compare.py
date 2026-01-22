from collections.abc import Collection
from collections.abc import Sized
from decimal import Decimal
import math
from numbers import Complex
import pprint
from types import TracebackType
from typing import Any
from typing import Callable
from typing import cast
from typing import ContextManager
from typing import final
from typing import List
from typing import Mapping
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import _pytest._code
from _pytest.outcomes import fail
def _repr_compare(self, other_side: Sequence[float]) -> List[str]:
    import math
    if len(self.expected) != len(other_side):
        return ['Impossible to compare lists with different sizes.', f'Lengths: {len(self.expected)} and {len(other_side)}']
    approx_side_as_map = _recursive_sequence_map(self._approx_scalar, self.expected)
    number_of_elements = len(approx_side_as_map)
    max_abs_diff = -math.inf
    max_rel_diff = -math.inf
    different_ids = []
    for i, (approx_value, other_value) in enumerate(zip(approx_side_as_map, other_side)):
        if approx_value != other_value:
            abs_diff = abs(approx_value.expected - other_value)
            max_abs_diff = max(max_abs_diff, abs_diff)
            if other_value == 0.0:
                max_rel_diff = math.inf
            else:
                max_rel_diff = max(max_rel_diff, abs_diff / abs(other_value))
            different_ids.append(i)
    message_data = [(str(i), str(other_side[i]), str(approx_side_as_map[i])) for i in different_ids]
    return _compare_approx(self.expected, message_data, number_of_elements, different_ids, max_abs_diff, max_rel_diff)