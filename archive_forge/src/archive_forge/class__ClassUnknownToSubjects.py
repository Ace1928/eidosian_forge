import collections
from typing import Any, Callable, List, Tuple, Union
import itertools
class _ClassUnknownToSubjects:
    """Equality methods should be able to deal with the unexpected."""

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _ClassUnknownToSubjects)

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __hash__(self):
        return hash(_ClassUnknownToSubjects)