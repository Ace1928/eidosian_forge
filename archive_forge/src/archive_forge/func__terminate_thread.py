import ctypes
import logging
import os
import queue
import socket
import threading
import time
import traceback
import wandb
from wandb import wandb_sdk
from wandb.apis import InternalApi
from wandb.sdk.launch.sweeps import utils as sweep_utils
def _terminate_thread(thread):
    if not thread.is_alive():
        return
    if hasattr(thread, '_terminated'):
        return
    thread._terminated = True
    tid = getattr(thread, '_thread_id', None)
    if tid is None:
        for k, v in threading._active.items():
            if v is thread:
                tid = k
    if tid is None:
        return
    logger.debug(f'Terminating thread: {tid}')
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(Exception))
    if res == 0:
        return
    elif res != 1:
        logger.debug(f'Termination failed for thread {tid}')
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)