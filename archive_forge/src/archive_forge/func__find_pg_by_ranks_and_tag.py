import itertools
import collections.abc
import contextlib
import hashlib
import io
import logging
import os
import pickle
import sys
import time
import warnings
from collections import namedtuple
from datetime import timedelta
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import torch
from torch._C._distributed_c10d import (
from .constants import default_pg_timeout, default_pg_nccl_timeout
from .c10d_logger import _exception_logger, _time_logger
from .rendezvous import register_rendezvous_handler, rendezvous  # noqa: F401
def _find_pg_by_ranks_and_tag(tag: str, ranks: List[int]) -> ProcessGroup:
    if len(tag) > 0 and (not tag.startswith('ptd:')) and (not tag.startswith('user:')):
        tag = f'user:{tag}'
    for group in _world.tags_to_pg.get(tag, []):
        if group.size() != len(ranks):
            continue
        group_ranks = get_process_group_ranks(group)
        good = all((r in group_ranks for r in ranks))
        if good:
            return group
    return None