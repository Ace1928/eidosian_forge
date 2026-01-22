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
class _datetime_heap_agg(_heap_agg):

    def process(self, value):
        return format_date_time_sqlite(value)