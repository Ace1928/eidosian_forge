import itertools
from functools import partial
from typing import (
from ..util import minibatch, registry
def _batch_by_length(seqs: Sequence[Any], max_words: int, get_length=len) -> List[List[Any]]:
    """Given a list of sequences, return a batched list of indices into the
    list, where the batches are grouped by length, in descending order.

    Batches may be at most max_words in size, defined as max sequence length * size.
    """
    lengths_indices = [(get_length(seq), i) for i, seq in enumerate(seqs)]
    lengths_indices.sort()
    batches = []
    batch: List[int] = []
    for length, i in lengths_indices:
        if not batch:
            batch.append(i)
        elif length * (len(batch) + 1) <= max_words:
            batch.append(i)
        else:
            batches.append(batch)
            batch = [i]
    if batch:
        batches.append(batch)
    assert sum((len(b) for b in batches)) == len(seqs)
    batches = [list(sorted(batch)) for batch in batches]
    batches.reverse()
    return batches