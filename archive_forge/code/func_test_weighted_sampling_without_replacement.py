from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_weighted_sampling_without_replacement():
    population = range(4)
    p = [0.01, 0.33, 0.33, 0.33]
    k = 3
    sampled = random._weighted_sampling_without_replacement(population=population, weights=p, k=k)
    assert len(set(sampled)) == k