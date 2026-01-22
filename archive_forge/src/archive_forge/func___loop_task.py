from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
@staticmethod
def __loop_task(async_results, time_in_milliseconds=1, filter=None, callback=None):
    delay_seconds = time_in_milliseconds / 1000
    nvsmi = nvidia_smi.getInstance()
    while async_results.is_aborted() == False:
        results = nvsmi.DeviceQuery(filter)
        async_results.__last_results = results
        if callback is not None:
            callback(async_results, results)
        time.sleep(delay_seconds)