import math
from enum import Enum, unique
from typing import Dict, List, Sequence, Tuple, Union
def _flip_trace(trace: Tuple[_EditOperations, ...]) -> Tuple[_EditOperations, ...]:
    """Flip the trace of edit operations.

    Instead of rewriting a->b, get a recipe for rewriting b->a. Simply flips insertions and deletions.

    Args:
        trace: A tuple of edit operations.

    Return:
        inverted_trace:
            A tuple of inverted edit operations.

    """
    _flip_operations: Dict[_EditOperations, _EditOperations] = {_EditOperations.OP_INSERT: _EditOperations.OP_DELETE, _EditOperations.OP_DELETE: _EditOperations.OP_INSERT}

    def _replace_operation_or_retain(operation: _EditOperations, _flip_operations: Dict[_EditOperations, _EditOperations]) -> _EditOperations:
        if operation in _flip_operations:
            return _flip_operations.get(operation)
        return operation
    return tuple((_replace_operation_or_retain(operation, _flip_operations) for operation in trace))