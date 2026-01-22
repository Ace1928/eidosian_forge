from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __toString(val):
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return str(val)