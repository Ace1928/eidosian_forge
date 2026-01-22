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
class TestMemcacheClientOverrides(test_cache.BaseTestCase):

    def test_client_stripped_of_threading_local(self):
        """threading.local overrides are restored for _MemcacheClient"""
        client_class = _memcache_pool._MemcacheClient
        thread_local = client_class.__mro__[2]
        self.assertTrue(thread_local is threading.local)
        for field in thread_local.__dict__.keys():
            if field not in ('__dict__', '__weakref__'):
                self.assertNotEqual(id(getattr(thread_local, field, None)), id(getattr(client_class, field, None)))

    def test_can_create_with_kwargs(self):
        """Test for lp 1812935

        Note that in order to reproduce the bug, it is necessary to add the
        following to the top of oslo_cache/tests/__init__.py::

            import eventlet
            eventlet.monkey_patch()

        This should happen before any other imports in that file.
        """
        client = _memcache_pool._MemcacheClient('foo', check_keys=False)
        self.assertFalse(client.do_check_key)
        self.assertIsInstance(client, _memcache_pool._MemcacheClient)