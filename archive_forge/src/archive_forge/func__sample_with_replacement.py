from __future__ import annotations
import heapq
import math
import random as rnd
from functools import partial
from itertools import islice
from dask.bag.core import Bag
def _sample_with_replacement(population, k, split_every):
    return population.reduction(partial(_sample_with_replacement_map_partitions, k=k), partial(_sample_reduce, k=k, replace=True), out_type=Bag, split_every=split_every)