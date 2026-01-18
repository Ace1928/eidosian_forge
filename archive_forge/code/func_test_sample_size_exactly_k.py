from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_sample_size_exactly_k():
    seq = range(20)
    sut = db.from_sequence(seq, npartitions=3)
    li = list(random.sample(sut, k=2).compute())
    assert sut.map_partitions(len).compute() == (7, 7, 6)
    assert len(li) == 2
    assert all((i in seq for i in li))
    assert len(set(li)) == len(li)