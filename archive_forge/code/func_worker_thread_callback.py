import atexit
import traceback
import io
import socket, sys, threading
import posixpath
import time
import os
from itertools import count
import _thread
import queue
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import unquote, urlsplit
from paste.util import converters
import logging
def worker_thread_callback(self, message=None):
    """
        Worker thread should call this method to get and process queued
        callables.
        """
    thread_obj = threading.current_thread()
    thread_id = thread_obj.thread_id = _thread.get_ident()
    self.workers.append(thread_obj)
    self.idle_workers.append(thread_id)
    requests_processed = 0
    add_replacement_worker = False
    self.logger.debug('Started new worker %s: %s', thread_id, message)
    try:
        while True:
            if self.max_requests and self.max_requests < requests_processed:
                self.logger.debug('Thread %s processed %i requests (limit %s); stopping thread' % (thread_id, requests_processed, self.max_requests))
                add_replacement_worker = True
                break
            runnable = self.queue.get()
            if runnable is ThreadPool.SHUTDOWN:
                self.logger.debug('Worker %s asked to SHUTDOWN', thread_id)
                break
            try:
                self.idle_workers.remove(thread_id)
            except ValueError:
                pass
            self.worker_tracker[thread_id] = [time.time(), None]
            requests_processed += 1
            try:
                try:
                    runnable()
                except:
                    print('Unexpected exception in worker %r' % runnable, file=sys.stderr)
                    traceback.print_exc()
                if thread_id in self.dying_threads:
                    break
            finally:
                try:
                    del self.worker_tracker[thread_id]
                except KeyError:
                    pass
            self.idle_workers.append(thread_id)
    finally:
        try:
            del self.worker_tracker[thread_id]
        except KeyError:
            pass
        try:
            self.idle_workers.remove(thread_id)
        except ValueError:
            pass
        try:
            self.workers.remove(thread_obj)
        except ValueError:
            pass
        try:
            del self.dying_threads[thread_id]
        except KeyError:
            pass
        if add_replacement_worker:
            self.add_worker_thread(message='Voluntary replacement for thread %s' % thread_id)