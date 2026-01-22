import os
import numpy as np
import threading
from time import time
from .. import config, logging
def _get_num_threads(pid):
    """
    Function to get the number of threads a process is using

    Parameters
    ----------
    pid : integer
        the process ID of process to profile

    Returns
    -------
    num_threads : int
        the number of threads that the process is using

    """
    try:
        proc = psutil.Process(pid)
        if proc.status() == psutil.STATUS_RUNNING:
            num_threads = proc.num_threads()
        elif proc.num_threads() > 1:
            tprocs = [psutil.Process(thr.id) for thr in proc.threads()]
            alive_tprocs = [tproc for tproc in tprocs if tproc.status() == psutil.STATUS_RUNNING]
            num_threads = len(alive_tprocs)
        else:
            num_threads = 1
        child_threads = 0
        for child in proc.children(recursive=True):
            if len(child.children()) == 0:
                if child.status() == psutil.STATUS_RUNNING:
                    child_thr = child.num_threads()
                elif child.num_threads() > 1:
                    tprocs = [psutil.Process(thr.id) for thr in child.threads()]
                    alive_tprocs = [tproc for tproc in tprocs if tproc.status() == psutil.STATUS_RUNNING]
                    child_thr = len(alive_tprocs)
                else:
                    child_thr = 0
                child_threads += child_thr
    except psutil.NoSuchProcess:
        return None
    num_threads = max(child_threads, num_threads)
    return num_threads