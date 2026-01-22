import os
import re
import sys
import numpy as np
import inspect
import sysconfig
def _get_mem_available():
    """
    Get information about memory available, not counting swap.
    """
    try:
        import psutil
        return psutil.virtual_memory().available
    except (ImportError, AttributeError):
        pass
    if sys.platform.startswith('linux'):
        info = {}
        with open('/proc/meminfo') as f:
            for line in f:
                p = line.split()
                info[p[0].strip(':').lower()] = float(p[1]) * 1000.0
        if 'memavailable' in info:
            return info['memavailable']
        else:
            return info['memfree'] + info['cached']
    return None