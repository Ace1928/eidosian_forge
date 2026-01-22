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
def _silence_eof_received_warning(self):
    logger = logging.getLogger('asyncio')
    filter = logging.Filter('has no effect when using ssl')
    logger.addFilter(filter)
    try:
        yield
    finally:
        logger.removeFilter(filter)