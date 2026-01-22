import threading
import sys
from os.path import basename
from _pydev_bundle import pydev_log
from os import scandir
import time
@accept_directory.setter
def accept_directory(self, accept_directory):
    self._accept_directory = accept_directory
    for path_watcher in self._path_watchers:
        path_watcher.accept_directory = accept_directory