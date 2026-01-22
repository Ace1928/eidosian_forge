from __future__ import print_function
import logging
import os
import sys
import threading
import time
import subprocess
from wsgiref.simple_server import WSGIRequestHandler
from pecan.commands import BaseCommand
from pecan import util
class AggressiveEventHandler(FileSystemEventHandler):
    lock = threading.Lock()

    def should_reload(self, event):
        for t in (FileSystemMovedEvent, FileModifiedEvent, DirModifiedEvent):
            if isinstance(event, t):
                return True
        return False

    def on_modified(self, event):
        if self.should_reload(event) and self.lock.acquire(False):
            parent.server_process.kill()
            parent.create_subprocess()
            time.sleep(1)
            self.lock.release()