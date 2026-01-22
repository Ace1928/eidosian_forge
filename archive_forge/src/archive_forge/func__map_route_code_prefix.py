import datetime
import email.message
import math
from operator import methodcaller
import sys
import unittest
import warnings
from testtools.compat import _b
from testtools.content import (
from testtools.content_type import ContentType
from testtools.tags import TagContext
def _map_route_code_prefix(self, sink, route_prefix, consume_route=False):
    if '/' in route_prefix:
        raise TypeError(f'{route_prefix!r} is more than one route step long')
    self._route_code_prefixes[route_prefix] = (sink, consume_route)