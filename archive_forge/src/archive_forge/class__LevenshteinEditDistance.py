import math
from enum import Enum, unique
from typing import Dict, List, Sequence, Tuple, Union
class _LevenshteinEditDistance:
    """A convenience class for calculating the Levenshtein edit distance.

    Class will cache some intermediate values to hasten the calculation. The implementation follows the implementation
    from https://github.com/mjpost/sacrebleu/blob/master/sacrebleu/metrics/lib_ter.py,
    where the most of this implementation is adapted and copied from.

    Args:
        reference_tokens: list of reference tokens
        op_insert: cost of insertion operation
        op_delete: cost of deletion operation
        op_substitute: cost of substitution operation

    """

    def __init__(self, reference_tokens: List[str], op_insert: int=1, op_delete: int=1, op_substitute: int=1) -> None:
        self.reference_tokens = reference_tokens
        self.reference_len = len(reference_tokens)
        self.cache: Dict[str, Tuple[int, str]] = {}
        self.cache_size = 0
        self.op_insert = op_insert
        self.op_delete = op_delete
        self.op_substitute = op_substitute
        self.op_nothing = 0
        self.op_undefined = _INT_INFINITY

    def __call__(self, prediction_tokens: List[str]) -> Tuple[int, Tuple[_EditOperations, ...]]:
        """Calculate edit distance between self._words_ref and the hypothesis. Uses cache to skip some computations.

        Args:
            prediction_tokens: A tokenized predicted sentence.

        Return:
            A tuple of a calculated edit distance and a trace of executed operations.

        """
        start_position, cached_edit_distance = self._find_cache(prediction_tokens)
        edit_distance_int, edit_distance, trace = self._levenshtein_edit_distance(prediction_tokens, start_position, cached_edit_distance)
        self._add_cache(prediction_tokens, edit_distance)
        return (edit_distance_int, trace)

    def _levenshtein_edit_distance(self, prediction_tokens: List[str], prediction_start: int, cache: List[List[Tuple[int, _EditOperations]]]) -> Tuple[int, List[List[Tuple[int, _EditOperations]]], Tuple[_EditOperations, ...]]:
        """Dynamic programming algorithm to compute the Levenhstein edit distance.

        Args:
            prediction_tokens: A tokenized predicted sentence.
            prediction_start: An index where a predicted sentence to be considered from.
            cache: A cached Levenshtein edit distance.

        Returns:
            Edit distance between the predicted sentence and the reference sentence

        """
        prediction_len = len(prediction_tokens)
        empty_rows: List[List[Tuple[int, _EditOperations]]] = [list(self._get_empty_row(self.reference_len)) for _ in range(prediction_len - prediction_start)]
        edit_distance: List[List[Tuple[int, _EditOperations]]] = cache + empty_rows
        length_ratio = self.reference_len / prediction_len if prediction_tokens else 1.0
        beam_width = math.ceil(length_ratio / 2 + _BEAM_WIDTH) if length_ratio / 2 > _BEAM_WIDTH else _BEAM_WIDTH
        for i in range(prediction_start + 1, prediction_len + 1):
            pseudo_diag = math.floor(i * length_ratio)
            min_j = max(0, pseudo_diag - beam_width)
            max_j = self.reference_len + 1 if i == prediction_len else min(self.reference_len + 1, pseudo_diag + beam_width)
            for j in range(min_j, max_j):
                if j == 0:
                    edit_distance[i][j] = (edit_distance[i - 1][j][0] + self.op_delete, _EditOperations.OP_DELETE)
                else:
                    if prediction_tokens[i - 1] == self.reference_tokens[j - 1]:
                        cost_substitute = self.op_nothing
                        operation_substitute = _EditOperations.OP_NOTHING
                    else:
                        cost_substitute = self.op_substitute
                        operation_substitute = _EditOperations.OP_SUBSTITUTE
                    operations = ((edit_distance[i - 1][j - 1][0] + cost_substitute, operation_substitute), (edit_distance[i - 1][j][0] + self.op_delete, _EditOperations.OP_DELETE), (edit_distance[i][j - 1][0] + self.op_insert, _EditOperations.OP_INSERT))
                    for operation_cost, operation_name in operations:
                        if edit_distance[i][j][0] > operation_cost:
                            edit_distance[i][j] = (operation_cost, operation_name)
        trace = self._get_trace(prediction_len, edit_distance)
        return (edit_distance[-1][-1][0], edit_distance[len(cache):], trace)

    def _get_trace(self, prediction_len: int, edit_distance: List[List[Tuple[int, _EditOperations]]]) -> Tuple[_EditOperations, ...]:
        """Get a trace of executed operations from the edit distance matrix.

        Args:
            prediction_len: A length of a tokenized predicted sentence.
            edit_distance:
                A matrix of the Levenshtedin edit distance. The element part of the matrix is a tuple of an edit
                operation cost and an edit operation itself.

        Return:
            A trace of executed operations returned as a tuple of `_EDIT_OPERATIONS` enumerates.

        Raises:
            ValueError:
                If an unknown operation has been applied.

        """
        trace: Tuple[_EditOperations, ...] = ()
        i = prediction_len
        j = self.reference_len
        while i > 0 or j > 0:
            operation = edit_distance[i][j][1]
            trace = (operation, *trace)
            if operation in (_EditOperations.OP_SUBSTITUTE, _EditOperations.OP_NOTHING):
                i -= 1
                j -= 1
            elif operation == _EditOperations.OP_INSERT:
                j -= 1
            elif operation == _EditOperations.OP_DELETE:
                i -= 1
            else:
                raise ValueError(f'Unknown operation {operation!r}')
        return trace

    def _add_cache(self, prediction_tokens: List[str], edit_distance: List[List[Tuple[int, _EditOperations]]]) -> None:
        """Add newly computed rows to cache.

        Since edit distance is only calculated on the hypothesis suffix that was not in cache, the number of rows in
        `edit_distance` matrx may be shorter than hypothesis length. In that case we skip over these initial words.

        Args:
            prediction_tokens: A tokenized predicted sentence.
            edit_distance:
                A matrix of the Levenshtedin edit distance. The element part of the matrix is a tuple of an edit
                operation cost and an edit operation itself.

        """
        if self.cache_size >= _MAX_CACHE_SIZE:
            return
        node = self.cache
        skip_num = len(prediction_tokens) - len(edit_distance)
        for i in range(skip_num):
            node = node[prediction_tokens[i]][0]
        for word, row in zip(prediction_tokens[skip_num:], edit_distance):
            if word not in node:
                node[word] = ({}, tuple(row))
                self.cache_size += 1
            value = node[word]
            node = value[0]

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

    def _get_empty_row(self, length: int) -> List[Tuple[int, _EditOperations]]:
        """Precomputed empty matrix row for Levenhstein edit distance.

        Args:
            length: A length of a tokenized sentence.

        Return:
            A list of tuples containing infinite edit operation costs and yet undefined edit operations.

        """
        return [(int(self.op_undefined), _EditOperations.OP_UNDEFINED)] * (length + 1)

    def _get_initial_row(self, length: int) -> List[Tuple[int, _EditOperations]]:
        """First row corresponds to insertion operations of the reference, so 1 edit operation per reference word.

        Args:
            length: A length of a tokenized sentence.

        Return:
            A list of tuples containing edit operation costs of insert and insert edit operations.

        """
        return [(i * self.op_insert, _EditOperations.OP_INSERT) for i in range(length + 1)]