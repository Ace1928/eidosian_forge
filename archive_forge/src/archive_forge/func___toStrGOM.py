from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __toStrGOM(mode):
    if mode == NVML_GOM_ALL_ON:
        return 'All On'
    elif mode == NVML_GOM_COMPUTE:
        return 'Compute'
    elif mode == NVML_GOM_LOW_DP:
        return 'Low Double Precision'
    else:
        return 'Unknown'