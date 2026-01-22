import time
import logging
import datetime
import itertools
import functools
import threading
from pyzor.engines.common import *
def _release_connection(self, db):
    if self.bound:
        self.db_queue.put(db)
    else:
        db.close()