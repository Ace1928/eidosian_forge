import collections
import logging
import time
from functools import partial
def _on_update(self, workername, method, response):
    info = self.workers[workername]
    info[method] = response
    info['timestamp'] = time.time()