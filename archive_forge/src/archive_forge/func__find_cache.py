import math
from enum import Enum, unique
from typing import Dict, List, Sequence, Tuple, Union
def _find_cache(self, prediction_tokens: List[str]) -> Tuple[int, List[List[Tuple[int, _EditOperations]]]]:
    """Find the already calculated rows of the Levenshtein edit distance metric.

        Args:
            prediction_tokens: A tokenized predicted sentence.

        Return:
            A tuple of a start hypothesis position and `edit_distance` matrix.

            prediction_start: An index where a predicted sentence to be considered from.
            edit_distance:
                A matrix of the cached Levenshtedin edit distance. The element part of the matrix is a tuple of an edit
                operation cost and an edit operation itself.

        """
    node = self.cache
    start_position = 0
    edit_distance: List[List[Tuple[int, _EditOperations]]] = [self._get_initial_row(self.reference_len)]
    for word in prediction_tokens:
        if word in node:
            start_position += 1
            node, row = node[word]
            edit_distance.append(row)
        else:
            break
    return (start_position, edit_distance)