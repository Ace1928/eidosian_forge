import threading
import sys
from os.path import basename
from _pydev_bundle import pydev_log
from os import scandir
import time
class _SingleVisitInfo(object):

    def __init__(self):
        self.count = 0
        self.visited_dirs = set()
        self.file_to_mtime = {}
        self.last_sleep_time = time.time()