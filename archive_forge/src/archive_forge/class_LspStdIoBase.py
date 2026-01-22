import asyncio
import io
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Text
from tornado.concurrent import run_on_executor
from tornado.gen import convert_yielded
from tornado.httputil import HTTPHeaders
from tornado.ioloop import IOLoop
from tornado.queues import Queue
from traitlets import Float, Instance, default
from traitlets.config import LoggingConfigurable
from .non_blocking import make_non_blocking
class LspStdIoBase(LoggingConfigurable):
    """Non-blocking, queued base for communicating with stdio Language Servers"""
    executor = None
    stream = Instance(io.RawIOBase, help='the stream to read/write')
    queue = Instance(Queue, help='queue to get/put')

    def __repr__(self):
        return '<{}(parent={})>'.format(self.__class__.__name__, self.parent)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log.debug('%s initialized', self)
        self.executor = ThreadPoolExecutor(max_workers=1)

    def close(self):
        self.stream.close()
        self.log.debug('%s closed', self)