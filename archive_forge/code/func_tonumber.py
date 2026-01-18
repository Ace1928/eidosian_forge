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
@udf(MATH)
def tonumber(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except:
            return None