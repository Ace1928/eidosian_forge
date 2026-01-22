from typing import Dict, FrozenSet, Generic, Iterable, Mapping, Optional, Set, TypeVar
from ortools.math_opt import sparse_containers_pb2
from ortools.math_opt.python import model
class SparseVectorFilter(Generic[VarOrConstraintType]):
    """Restricts the variables or constraints returned in a sparse vector.

    The default behavior is to return entries for all variables/constraints.

    E.g. when requesting the solution to an optimization problem, use this class
    to restrict the variables that values are returned for.

    Attributes:
      skip_zero_values: Do not include key value pairs with value zero.
      filtered_items: If not None, include only key value pairs these keys. Note
        that the empty set is different (don't return any keys) from None (return
        all keys).
    """

    def __init__(self, *, skip_zero_values: bool=False, filtered_items: Optional[Iterable[VarOrConstraintType]]=None):
        self._skip_zero_values: bool = skip_zero_values
        self._filtered_items: Optional[Set[VarOrConstraintType]] = None if filtered_items is None else frozenset(filtered_items)

    @property
    def skip_zero_values(self) -> bool:
        return self._skip_zero_values

    @property
    def filtered_items(self) -> Optional[FrozenSet[VarOrConstraintType]]:
        return self._filtered_items

    def to_proto(self):
        """Returns an equivalent proto representation."""
        result = sparse_containers_pb2.SparseVectorFilterProto()
        result.skip_zero_values = self._skip_zero_values
        if self._filtered_items is not None:
            result.filter_by_ids = True
            result.filtered_ids[:] = sorted((t.id for t in self._filtered_items))
        return result