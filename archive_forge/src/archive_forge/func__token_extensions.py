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
def _token_extensions(self):
    """
        Return a string representation of the OPTIONAL key-value pairs
        that can be sent with an OAUTHBEARER initial request.
        """
    if callable(getattr(self._sasl_oauth_token_provider, 'extensions', None)):
        extensions = self._sasl_oauth_token_provider.extensions()
        if len(extensions) > 0:
            msg = '\x01'.join([f'{k}={v}' for k, v in extensions.items()])
            return '\x01' + msg
    return ''