from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_sample_k_bigger_than_bag_size():
    seq = range(3)
    sut = db.from_sequence(seq, npartitions=3)
    with pytest.raises(ValueError, match='Sample larger than population'):
        random.sample(sut, k=4).compute()