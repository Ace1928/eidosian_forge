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
def _last_started_strategy(cb, started_at, finished_at, metrics):
    how_often = cb._periodic_spacing
    return started_at + how_often