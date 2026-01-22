import math
from enum import Enum, unique
from typing import Dict, List, Sequence, Tuple, Union
def _trace_to_alignment(trace: Tuple[_EditOperations, ...]) -> Tuple[Dict[int, int], List[int], List[int]]:
    """Transform trace of edit operations into an alignment of the sequences.

    Args:
        trace: A trace of edit operations as a tuple of `_EDIT_OPERATIONS` enumerates.

    Return:
        alignments: A dictionary mapping aligned positions between a reference and a hypothesis.
        reference_errors: A list of error positions in a reference.
        hypothesis_errors: A list of error positions in a hypothesis.

    Raises:
        ValueError:
            If an unknown operation is

    """
    reference_position = hypothesis_position = -1
    reference_errors: List[int] = []
    hypothesis_errors: List[int] = []
    alignments: Dict[int, int] = {}
    for operation in trace:
        if operation == _EditOperations.OP_NOTHING:
            hypothesis_position += 1
            reference_position += 1
            alignments[reference_position] = hypothesis_position
            reference_errors.append(0)
            hypothesis_errors.append(0)
        elif operation == _EditOperations.OP_SUBSTITUTE:
            hypothesis_position += 1
            reference_position += 1
            alignments[reference_position] = hypothesis_position
            reference_errors.append(1)
            hypothesis_errors.append(1)
        elif operation == _EditOperations.OP_INSERT:
            hypothesis_position += 1
            hypothesis_errors.append(1)
        elif operation == _EditOperations.OP_DELETE:
            reference_position += 1
            alignments[reference_position] = hypothesis_position
            reference_errors.append(1)
        else:
            raise ValueError(f'Unknown operation {operation!r}.')
    return (alignments, reference_errors, hypothesis_errors)