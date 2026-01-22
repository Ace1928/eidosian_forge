import collections
import itertools
import json
import random
from threading import Lock
from threading import Thread
import time
import uuid
import pytest
from dogpile.cache import CacheRegion
from dogpile.cache import register_backend
from dogpile.cache.api import CacheBackend
from dogpile.cache.api import CacheMutex
from dogpile.cache.api import CantDeserializeException
from dogpile.cache.api import NO_VALUE
from dogpile.cache.region import _backend_loader
from .assertions import assert_raises_message
from .assertions import eq_
class _GenericMutexTestSuite(_GenericBackendFixture):

    def test_mutex(self):
        backend = self._backend()
        mutex = backend.get_mutex('foo')
        assert not mutex.locked()
        ac = mutex.acquire()
        assert ac
        ac2 = mutex.acquire(False)
        assert mutex.locked()
        assert not ac2
        mutex.release()
        assert not mutex.locked()
        ac3 = mutex.acquire()
        assert ac3
        mutex.release()

    def test_subclass_match(self):
        backend = self._backend()
        mutex = backend.get_mutex('foo')
        assert isinstance(mutex, CacheMutex)

    @pytest.mark.time_intensive
    def test_mutex_threaded(self):
        backend = self._backend()
        backend.get_mutex('foo')
        lock = Lock()
        canary = []

        def f():
            for x in range(5):
                mutex = backend.get_mutex('foo')
                mutex.acquire()
                for y in range(5):
                    ack = lock.acquire(False)
                    canary.append(ack)
                    time.sleep(0.002)
                    if ack:
                        lock.release()
                mutex.release()
                time.sleep(0.02)
        threads = [Thread(target=f) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert False not in canary

    def test_mutex_reentrant_across_keys(self):
        backend = self._backend()
        for x in range(3):
            m1 = backend.get_mutex('foo')
            m2 = backend.get_mutex('bar')
            try:
                m1.acquire()
                assert m2.acquire(False)
                assert not m2.acquire(False)
                m2.release()
                assert m2.acquire(False)
                assert not m2.acquire(False)
                m2.release()
            finally:
                m1.release()

    def test_reentrant_dogpile(self):
        reg = self._region()

        def create_foo():
            return 'foo' + reg.get_or_create('bar', create_bar)

        def create_bar():
            return 'bar'
        eq_(reg.get_or_create('foo', create_foo), 'foobar')
        eq_(reg.get_or_create('foo', create_foo), 'foobar')