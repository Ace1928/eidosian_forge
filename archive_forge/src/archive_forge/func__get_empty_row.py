import math
from enum import Enum, unique
from typing import Dict, List, Sequence, Tuple, Union
def _get_empty_row(self, length: int) -> List[Tuple[int, _EditOperations]]:
    """Precomputed empty matrix row for Levenhstein edit distance.

        Args:
            length: A length of a tokenized sentence.

        Return:
            A list of tuples containing infinite edit operation costs and yet undefined edit operations.

        """
    return [(int(self.op_undefined), _EditOperations.OP_UNDEFINED)] * (length + 1)