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
class BaseSaslAuthenticator:

    def step(self, payload):
        return self._loop.run_in_executor(None, self._step, payload)

    def _step(self, payload):
        """ Process next token in sequence and return with:
            ``None`` if it was the last needed exchange
            ``tuple`` tuple with new token and a boolean whether it requires an
                answer token
        """
        try:
            data = self._authenticator.send(payload)
        except StopIteration:
            return
        else:
            return data