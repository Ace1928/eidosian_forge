import copy
import os
import pickle
import warnings
import numpy as np
@staticmethod
def _readMeta(fd):
    """Read meta array from the top of a file. Read lines until a blank line is reached.
        This function should ideally work for ALL versions of MetaArray.
        """
    meta = u''
    while True:
        line = fd.readline().strip()
        if line == '':
            break
        meta += line
    ret = eval(meta)
    return ret