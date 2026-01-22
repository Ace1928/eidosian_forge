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
def _compare_approx(full_object: object, message_data: Sequence[Tuple[str, str, str]], number_of_elements: int, different_ids: Sequence[object], max_abs_diff: float, max_rel_diff: float) -> List[str]:
    message_list = list(message_data)
    message_list.insert(0, ('Index', 'Obtained', 'Expected'))
    max_sizes = [0, 0, 0]
    for index, obtained, expected in message_list:
        max_sizes[0] = max(max_sizes[0], len(index))
        max_sizes[1] = max(max_sizes[1], len(obtained))
        max_sizes[2] = max(max_sizes[2], len(expected))
    explanation = [f'comparison failed. Mismatched elements: {len(different_ids)} / {number_of_elements}:', f'Max absolute difference: {max_abs_diff}', f'Max relative difference: {max_rel_diff}'] + [f'{indexes:<{max_sizes[0]}} | {obtained:<{max_sizes[1]}} | {expected:<{max_sizes[2]}}' for indexes, obtained, expected in message_list]
    return explanation