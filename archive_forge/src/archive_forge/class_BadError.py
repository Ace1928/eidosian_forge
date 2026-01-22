from __future__ import absolute_import
import mock
import pytest
from urllib3 import HTTPConnectionPool
from urllib3.exceptions import EmptyPoolError
from urllib3.packages.six.moves import queue
class BadError(Exception):
    """
    This should not be raised.
    """
    pass