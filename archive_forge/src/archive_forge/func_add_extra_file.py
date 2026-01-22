import os
import os.path
import re
import sys
import time
import threading
def add_extra_file(self, filename):
    dirname = os.path.dirname(filename)
    if dirname in self._dirs:
        return
    self._watcher.add_watch(dirname, mask=self.event_mask)
    self._dirs.add(dirname)