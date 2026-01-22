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
class _heap_agg(object):

    def __init__(self):
        self.heap = []
        self.ct = 0

    def process(self, value):
        return value

    def step(self, value):
        self.ct += 1
        heapq.heappush(self.heap, self.process(value))