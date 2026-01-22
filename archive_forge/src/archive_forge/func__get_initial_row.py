import math
from enum import Enum, unique
from typing import Dict, List, Sequence, Tuple, Union
def _get_initial_row(self, length: int) -> List[Tuple[int, _EditOperations]]:
    """First row corresponds to insertion operations of the reference, so 1 edit operation per reference word.

        Args:
            length: A length of a tokenized sentence.

        Return:
            A list of tuples containing edit operation costs of insert and insert edit operations.

        """
    return [(i * self.op_insert, _EditOperations.OP_INSERT) for i in range(length + 1)]