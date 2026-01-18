from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def test_reservoir_sample_map_partitions_correctness():
    N, k = (20, 10)
    seq = list(range(N))
    distribution = [0 for _ in range(N)]
    expected_distribution = [0 for _ in range(N)]
    reps = 2000
    for _ in range(reps):
        picks, _ = random._sample_map_partitions(seq, k)
        for pick in picks:
            distribution[pick] += 1
        for pick in rnd.sample(seq, k=k):
            expected_distribution[pick] += 1
    distribution = [c / (reps * k) for c in distribution]
    expected_distribution = [c / (reps * k) for c in expected_distribution]
    assert math.isclose(0.0, bhattacharyya(distribution, expected_distribution), abs_tol=0.01)