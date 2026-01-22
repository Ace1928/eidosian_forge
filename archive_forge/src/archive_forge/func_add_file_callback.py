import os
import sys
import time
import threading
import traceback
from paste.util.classinstance import classinstancemethod
def add_file_callback(self, cls, callback):
    """Add a callback -- a function that takes no parameters -- that will
        return a list of filenames to watch for changes."""
    if self is None:
        for instance in cls.instances:
            instance.add_file_callback(callback)
        cls.global_file_callbacks.append(callback)
    else:
        self.file_callbacks.append(callback)