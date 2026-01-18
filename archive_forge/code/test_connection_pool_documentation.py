import queue
import threading
import time
from unittest import mock
import testtools
from testtools import matchers
from oslo_cache import _bmemcache_pool
from oslo_cache import _memcache_pool
from oslo_cache import exception
from oslo_cache.tests import test_cache
Test for lp 1812935

        Note that in order to reproduce the bug, it is necessary to add the
        following to the top of oslo_cache/tests/__init__.py::

            import eventlet
            eventlet.monkey_patch()

        This should happen before any other imports in that file.
        