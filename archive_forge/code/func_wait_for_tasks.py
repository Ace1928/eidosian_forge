import time as _time
from sys import exit as sysexit
from os import _exit as osexit
from threading import Thread, Semaphore
from multiprocessing import Process, cpu_count
def wait_for_tasks(sleep=0):
    config['KILL_RECEIVED'] = True
    if config['POOLS'][config['POOL_NAME']]['threads'] == 0:
        return True
    try:
        while True:
            running = len([t.join(1) for t in config['TASKS'] if t is not None and t.is_alive()])
            if running == 0:
                break
            _time.sleep(sleep)
    except Exception:
        pass
    config['KILL_RECEIVED'] = False
    return True