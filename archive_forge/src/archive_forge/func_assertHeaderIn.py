from your HTTP server.
import pprint
import re
import socket
import sys
import time
import traceback
import os
import json
import unittest  # pylint: disable=deprecated-module,preferred-module
import warnings
import functools
import http.client
import urllib.parse
from more_itertools.more import always_iterable
import jaraco.functools
def assertHeaderIn(self, key, values, msg=None):
    """Fail if header indicated by key doesn't have one of the values."""
    lowkey = key.lower()
    for k, v in self.headers:
        if k.lower() == lowkey:
            matches = [value for value in values if str(value) == v]
            if matches:
                return matches
    if msg is None:
        msg = '%(key)r not in %(values)r' % vars()
    self._handlewebError(msg)