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
def _now_plus_periodicity(cb, now):
    how_often = cb._periodic_spacing
    return how_often + now