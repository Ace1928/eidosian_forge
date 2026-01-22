import os
import numpy as np
import threading
from time import time
from .. import config, logging
def _get_ram_mb(pid, pyfunc=False):
    """
    Function to get the RAM usage of a process and its children
    Reference: http://ftp.dev411.com/t/python/python-list/095thexx8g/multiprocessing-forking-memory-usage

    Parameters
    ----------
    pid : integer
        the PID of the process to get RAM usage of
    pyfunc : boolean (optional); default=False
        a flag to indicate if the process is a python function;
        when Pythons are multithreaded via multiprocess or threading,
        children functions include their own memory + parents. if this
        is set, the parent memory will removed from children memories


    Returns
    -------
    mem_mb : float
        the memory RAM in MB utilized by the process PID

    """
    try:
        parent = psutil.Process(pid)
        parent_mem = parent.memory_info().rss
        mem_mb = parent_mem / _MB
        for child in parent.children(recursive=True):
            child_mem = child.memory_info().rss
            if pyfunc:
                child_mem -= parent_mem
            mem_mb += child_mem / _MB
    except psutil.NoSuchProcess:
        return None
    return mem_mb