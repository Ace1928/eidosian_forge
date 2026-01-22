import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
def _whitelist(self, keys, db=None):
    c = db.cursor()
    try:
        c.executemany('INSERT INTO %s (digest, r_count, wl_count, r_entered, r_updated, wl_entered, wl_updated) VALUES (%%s, 0, 1, NOW(), NOW(), NOW(), NOW()) ON DUPLICATE KEY UPDATE wl_count=wl_count+1, wl_updated=NOW()' % self.table_name, itertools.imap(lambda key: (key,), keys))
    finally:
        c.close()