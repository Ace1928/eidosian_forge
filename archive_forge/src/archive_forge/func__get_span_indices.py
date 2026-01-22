from typing import Callable, List, Tuple
from thinc.api import Model, to_numpy
from thinc.types import Ints1d, Ragged
from ..util import registry
def _get_span_indices(ops, spans: Ragged, lengths: Ints1d) -> Ints1d:
    """Construct a flat array that has the indices we want to extract from the
    source data. For instance, if we want the spans (5, 9), (8, 10) the
    indices will be [5, 6, 7, 8, 8, 9].
    """
    spans, lengths = _ensure_cpu(spans, lengths)
    indices: List[int] = []
    offset = 0
    for i, length in enumerate(lengths):
        spans_i = spans[i].dataXd + offset
        for j in range(spans_i.shape[0]):
            indices.extend(range(spans_i[j, 0], spans_i[j, 1]))
        offset += length
    return ops.asarray1i(indices)