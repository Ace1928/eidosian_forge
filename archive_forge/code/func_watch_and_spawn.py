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
def watch_and_spawn(self, conf):
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemMovedEvent, FileModifiedEvent, DirModifiedEvent
    print('Monitoring for changes...')
    self.create_subprocess()
    parent = self

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
    paths = self.paths_to_monitor(conf)
    event_handler = AggressiveEventHandler()
    for path, recurse in paths:
        observer = Observer()
        observer.schedule(event_handler, path=path, recursive=recurse)
        observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass