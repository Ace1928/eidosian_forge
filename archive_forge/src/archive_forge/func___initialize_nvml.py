from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __initialize_nvml():
    """ Initialize NVML bindings. """
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    handles = {}
    for i in range(0, deviceCount):
        handles[i] = nvmlDeviceGetHandleByIndex(i)
    return handles