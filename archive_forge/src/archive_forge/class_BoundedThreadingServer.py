import os
import sys
import time
import errno
import socket
import signal
import logging
import threading
import traceback
import email.message
import pyzor.config
import pyzor.account
import pyzor.engines.common
import pyzor.hacks.py26
class BoundedThreadingServer(ThreadingServer):
    """Same as ThreadingServer but this also accepts a limited number of
    concurrent threads.
    """

    def __init__(self, address, database, passwd_fn, access_fn, max_threads, forwarding_server=None):
        ThreadingServer.__init__(self, address, database, passwd_fn, access_fn, forwarder=forwarding_server)
        self.semaphore = threading.Semaphore(max_threads)

    def process_request(self, request, client_address):
        self.semaphore.acquire()
        ThreadingServer.process_request(self, request, client_address)

    def process_request_thread(self, request, client_address):
        ThreadingServer.process_request_thread(self, request, client_address)
        self.semaphore.release()