import datetime
import hashlib
import heapq
import math
import os
import random
import re
import sys
import threading
import zlib
from peewee import format_date_time
@aggregate(DATE)
class mintdiff(_datetime_heap_agg):

    def finalize(self):
        dtp = min_diff = None
        while self.heap:
            if min_diff is None:
                if dtp is None:
                    dtp = heapq.heappop(self.heap)
                    continue
            dt = heapq.heappop(self.heap)
            diff = dt - dtp
            if min_diff is None or min_diff > diff:
                min_diff = diff
            dtp = dt
        if min_diff is not None:
            return total_seconds(min_diff)