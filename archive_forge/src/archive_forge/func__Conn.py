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
@property
def _Conn(self):
    """Return HTTPConnection or HTTPSConnection based on self.scheme.

        * from :py:mod:`python:http.client`.
        """
    cls_name = '{scheme}Connection'.format(scheme=self.scheme.upper())
    return getattr(http.client, cls_name)