import asyncio
import asyncio.events
import collections
import contextlib
import gc
import logging
import os
import pprint
import re
import select
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
import uvloop
@contextlib.contextmanager
def unix_sock_name(self):
    with tempfile.TemporaryDirectory() as td:
        fn = os.path.join(td, 'sock')
        try:
            yield fn
        finally:
            try:
                os.unlink(fn)
            except OSError:
                pass