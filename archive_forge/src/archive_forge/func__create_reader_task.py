import asyncio
import collections
import base64
import functools
import hashlib
import hmac
import logging
import random
import socket
import struct
import sys
import time
import traceback
import uuid
import warnings
import weakref
import async_timeout
import aiokafka.errors as Errors
from aiokafka.abc import AbstractTokenProvider
from aiokafka.protocol.api import RequestHeader
from aiokafka.protocol.admin import (
from aiokafka.protocol.commit import (
from aiokafka.util import create_future, create_task, get_running_loop, wait_for
def _create_reader_task(self):
    self_ref = weakref.ref(self)
    read_task = create_task(self._read(self_ref))
    read_task.add_done_callback(functools.partial(self._on_read_task_error, self_ref))
    return read_task