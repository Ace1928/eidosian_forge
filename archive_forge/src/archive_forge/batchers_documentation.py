import itertools
from functools import partial
from typing import (
from ..util import minibatch, registry
Given a list of sequences, return a batched list of indices into the
    list, where the batches are grouped by length, in descending order.

    Batches may be at most max_words in size, defined as max sequence length * size.
    