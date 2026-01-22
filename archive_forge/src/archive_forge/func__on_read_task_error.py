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
@classmethod
def _on_read_task_error(cls, self_ref, read_task):
    if read_task.cancelled():
        return
    try:
        read_task.result()
    except Exception as exc:
        if not isinstance(exc, (OSError, EOFError, ConnectionError)):
            log.exception('Unexpected exception in AIOKafkaConnection')
        self = self_ref()
        if self is not None:
            self.close(reason=CloseReason.CONNECTION_BROKEN, exc=exc)