import collections
import random
import threading
import time
from concurrent import futures
import fasteners
from fasteners import test
from fasteners import _utils
def _find_overlaps(times, start, end):
    overlaps = 0
    for s, e in times:
        if s >= start and e <= end:
            overlaps += 1
    return overlaps