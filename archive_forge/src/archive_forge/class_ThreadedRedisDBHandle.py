import time
import logging
import datetime
import functools
from pyzor.engines.common import *
class ThreadedRedisDBHandle(RedisDBHandle):

    def __init__(self, fn, mode, max_age=None, bound=None):
        RedisDBHandle.__init__(self, fn, mode, max_age=max_age)