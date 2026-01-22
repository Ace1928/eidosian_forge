from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
class loop_async:
    __last_result = None
    __task = None
    __abort = False
    __callback_chain = None

    def __init__(self, time_in_milliseconds=1, filter=None, callback=None):
        self.__abort = False
        self.__callback_chain = callback
        self.__task = Thread(target=nvidia_smi.loop_async.__loop_task, args=(self, time_in_milliseconds, filter, nvidia_smi.loop_async.__callback))
        self.__task.start()

    def __del__(self):
        self.__abort = True
        self.__callback_chain = None

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

    def __callback(self, result):
        self.__last_result = result
        if self.__callback_chain is not None:
            self.__callback_chain(self, result)

    def cancel(self):
        self.__abort = True
        if self.__task is not None:
            self.__task.join()

    def is_aborted(self):
        return self.__abort

    def result(self):
        return self.__last_result