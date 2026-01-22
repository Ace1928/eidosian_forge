import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
def _really__setitem__(self, key, value, db=None):
    """__setitem__ without the exception handling."""
    c = db.cursor()
    try:
        c.execute('INSERT INTO %s (digest, r_count, wl_count, r_entered, r_updated, wl_entered, wl_updated) VALUES (%%s, %%s, %%s, %%s, %%s, %%s, %%s) ON DUPLICATE KEY UPDATE r_count=%%s, wl_count=%%s, r_entered=%%s, r_updated=%%s, wl_entered=%%s, wl_updated=%%s' % self.table_name, (key, value.r_count, value.wl_count, value.r_entered, value.r_updated, value.wl_entered, value.wl_updated, value.r_count, value.wl_count, value.r_entered, value.r_updated, value.wl_entered, value.wl_updated))
    finally:
        c.close()