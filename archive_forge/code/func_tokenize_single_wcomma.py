import re
import datetime
import numpy as np
import csv
import ctypes
def tokenize_single_wcomma(val):
    m = r_wcomattrval.match(val)
    if m:
        try:
            name = m.group(1).strip()
            type = m.group(2).strip()
        except IndexError as e:
            raise ValueError('Error while tokenizing attribute') from e
    else:
        raise ValueError('Error while tokenizing single %s' % val)
    return (name, type)