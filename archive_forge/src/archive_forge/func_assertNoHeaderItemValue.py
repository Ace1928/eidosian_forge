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
def assertNoHeaderItemValue(self, key, value, msg=None):
    """Fail if the header contains the specified value."""
    lowkey = key.lower()
    hdrs = self.headers
    matches = [k for k, v in hdrs if k.lower() == lowkey and v == value]
    if matches:
        if msg is None:
            msg = '%r:%r in %r' % (key, value, hdrs)
        self._handlewebError(msg)