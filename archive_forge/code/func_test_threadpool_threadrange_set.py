import os
import queue
import socket
import tempfile
import threading
import types
import uuid
import urllib.parse  # noqa: WPS301
import pytest
import requests
import requests_unixsocket
from pypytools.gc.custom import DefaultGc
from .._compat import bton, ntob
from .._compat import IS_LINUX, IS_MACOS, IS_WINDOWS, SYS_PLATFORM
from ..server import IS_UID_GID_RESOLVABLE, Gateway, HTTPServer
from ..workers.threadpool import ThreadPool
from ..testing import (
@pytest.mark.parametrize(('minthreads', 'maxthreads', 'inited_maxthreads'), ((1, -2, float('inf')), (1, -1, float('inf')), (1, 1, 1), (1, 2, 2), (1, float('inf'), float('inf')), (2, -2, float('inf')), (2, -1, float('inf')), (2, 2, 2), (2, float('inf'), float('inf'))))
def test_threadpool_threadrange_set(minthreads, maxthreads, inited_maxthreads):
    """Test setting the number of threads in a ThreadPool.

    The ThreadPool should properly set the min+max number of the threads to use
    in the pool if those limits are valid.
    """
    tp = ThreadPool(server=None, min=minthreads, max=maxthreads)
    assert tp.min == minthreads
    assert tp.max == inited_maxthreads