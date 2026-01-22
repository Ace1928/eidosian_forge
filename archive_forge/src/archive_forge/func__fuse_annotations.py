from __future__ import annotations
import itertools
import os
from collections.abc import Hashable, Iterable, Mapping, Sequence
from itertools import product
from math import prod
from typing import Any
import tlz as toolz
import dask
from dask.base import clone_key, get_name_from_key, tokenize
from dask.core import flatten, ishashable, keys_in_tasks, reverse_dict
from dask.highlevelgraph import HighLevelGraph, Layer
from dask.optimization import SubgraphCallable, fuse
from dask.typing import Graph, Key
from dask.utils import (
def _fuse_annotations(*args: dict) -> dict:
    """
    Given an iterable of annotations dictionaries, fuse them according
    to some simple rules.
    """
    annotations = toolz.merge(*args)
    retries = [a['retries'] for a in args if 'retries' in a]
    if retries:
        annotations['retries'] = max(retries)
    priorities = [a['priority'] for a in args if 'priority' in a]
    if priorities:
        annotations['priority'] = max(priorities)
    resources = [a['resources'] for a in args if 'resources' in a]
    if resources:
        annotations['resources'] = toolz.merge_with(max, *resources)
    workers = [a['workers'] for a in args if 'workers' in a]
    if workers:
        annotations['workers'] = list(set.intersection(*[set(w) for w in workers]))
    allow_other_workers = [a['allow_other_workers'] for a in args if 'allow_other_workers' in a]
    if allow_other_workers:
        annotations['allow_other_workers'] = all(allow_other_workers)
    return annotations