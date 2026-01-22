import datetime
import functools
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
import urllib
import uuid
import requests
import keystoneauth1
from keystoneauth1 import _utils as utils
from keystoneauth1 import discover
from keystoneauth1 import exceptions
class _Retries(object):
    __slots__ = ('_fixed_delay', '_current')

    def __init__(self, fixed_delay=None):
        self._fixed_delay = fixed_delay
        self.reset()

    def __next__(self):
        value = self._current
        if not self._fixed_delay:
            self._current = min(value * 2, _MAX_RETRY_INTERVAL)
        return value

    def reset(self):
        if self._fixed_delay:
            self._current = self._fixed_delay
        else:
            self._current = _EXPONENTIAL_DELAY_START
    next = __next__