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
class MockPattern(str):

    def __eq__(self, other):
        return bool(re.search(str(self), other, re.S))