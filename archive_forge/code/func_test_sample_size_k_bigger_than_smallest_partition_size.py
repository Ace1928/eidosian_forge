from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_sample_size_k_bigger_than_smallest_partition_size():
    seq = range(10)
    sut = db.from_sequence(seq, partition_size=9)
    li = list(random.sample(sut, k=2).compute())
    assert sut.map_partitions(len).compute() == (9, 1)
    assert len(li) == 2
    assert all((i in seq for i in li))
    assert len(set(li)) == len(li)