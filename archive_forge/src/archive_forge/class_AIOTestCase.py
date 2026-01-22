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
class AIOTestCase(BaseTestCase):
    implementation = 'asyncio'

    def setUp(self):
        super().setUp()
        if sys.version_info < (3, 12):
            watcher = asyncio.SafeChildWatcher()
            watcher.attach_loop(self.loop)
            asyncio.set_child_watcher(watcher)

    def tearDown(self):
        if sys.version_info < (3, 12):
            asyncio.set_child_watcher(None)
        super().tearDown()

    def new_loop(self):
        return asyncio.new_event_loop()

    def new_policy(self):
        return asyncio.DefaultEventLoopPolicy()