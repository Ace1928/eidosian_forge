import collections
import fractions
import functools
import heapq
import inspect
import logging
import math
import random
import threading
from concurrent import futures
import futurist
from futurist import _utils as utils
class _Schedule(object):
    """Internal heap-based structure that maintains the schedule/ordering.

    This stores a heap composed of the following ``(next_run, index)`` where
    ``next_run`` is the next desired runtime for the callback that is stored
    somewhere with the index provided. The index is saved so that if two
    functions with the same ``next_run`` time are inserted, that the one with
    the smaller index is preferred (it is also saved so that on pop we can
    know what the index of the callback we should call is).
    """

    def __init__(self):
        self._ordering = []

    def push(self, next_run, index):
        heapq.heappush(self._ordering, (next_run, index))

    def __len__(self):
        return len(self._ordering)

    def fetch_next_run(self, index):
        for next_run, a_index in self._ordering:
            if a_index == index:
                return next_run
        return None

    def pop(self):
        return heapq.heappop(self._ordering)