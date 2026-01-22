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
class _GenericBackendTestSuite(_GenericBackendFixture):

    def test_backend_get_nothing(self):
        backend = self._backend()
        some_key = gen_some_key()
        eq_(backend.get_serialized(some_key), NO_VALUE)

    def test_backend_delete_nothing(self):
        backend = self._backend()
        some_key = gen_some_key()
        backend.delete(some_key)

    def test_backend_set_get_value(self):
        backend = self._backend()
        some_key = gen_some_key()
        backend.set_serialized(some_key, b'some value')
        eq_(backend.get_serialized(some_key), b'some value')

    def test_backend_delete(self):
        backend = self._backend()
        some_key = gen_some_key()
        backend.set_serialized(some_key, b'some value')
        backend.delete(some_key)
        eq_(backend.get_serialized(some_key), NO_VALUE)

    def test_region_is_key_locked(self):
        reg = self._region()
        random_key = str(uuid.uuid1())
        assert not reg.get(random_key)
        eq_(reg.key_is_locked(random_key), False)
        eq_(reg.key_is_locked(random_key), False)
        mutex = reg.backend.get_mutex(random_key)
        if mutex:
            mutex.acquire()
            eq_(reg.key_is_locked(random_key), True)
            mutex.release()
            eq_(reg.key_is_locked(random_key), False)

    def test_region_set_get_value(self):
        reg = self._region()
        some_key = gen_some_key()
        reg.set(some_key, 'some value')
        eq_(reg.get(some_key), 'some value')

    def test_region_set_multiple_values(self):
        reg = self._region()
        values = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        reg.set_multi(values)
        eq_(values['key1'], reg.get('key1'))
        eq_(values['key2'], reg.get('key2'))
        eq_(values['key3'], reg.get('key3'))

    def test_region_get_zero_multiple_values(self):
        reg = self._region()
        eq_(reg.get_multi([]), [])

    def test_region_set_zero_multiple_values(self):
        reg = self._region()
        reg.set_multi({})

    def test_region_set_zero_multiple_values_w_decorator(self):
        reg = self._region()
        values = reg.get_or_create_multi([], lambda: 0)
        eq_(values, [])

    def test_region_get_or_create_multi_w_should_cache_none(self):
        reg = self._region()
        values = reg.get_or_create_multi(['key1', 'key2', 'key3'], lambda *k: [None, None, None], should_cache_fn=lambda v: v is not None)
        eq_(values, [None, None, None])

    def test_region_get_multiple_values(self):
        reg = self._region()
        key1 = 'value1'
        key2 = 'value2'
        key3 = 'value3'
        reg.set('key1', key1)
        reg.set('key2', key2)
        reg.set('key3', key3)
        values = reg.get_multi(['key1', 'key2', 'key3'])
        eq_([key1, key2, key3], values)

    def test_region_get_nothing_multiple(self):
        reg = self._region()
        reg.delete_multi(['key1', 'key2', 'key3', 'key4', 'key5'])
        values = {'key1': 'value1', 'key3': 'value3', 'key5': 'value5'}
        reg.set_multi(values)
        reg_values = reg.get_multi(['key1', 'key2', 'key3', 'key4', 'key5', 'key6'])
        eq_(reg_values, ['value1', NO_VALUE, 'value3', NO_VALUE, 'value5', NO_VALUE])

    def test_region_get_empty_multiple(self):
        reg = self._region()
        reg_values = reg.get_multi([])
        eq_(reg_values, [])

    def test_region_delete_multiple(self):
        reg = self._region()
        values = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
        reg.set_multi(values)
        reg.delete_multi(['key2', 'key10'])
        eq_(values['key1'], reg.get('key1'))
        eq_(NO_VALUE, reg.get('key2'))
        eq_(values['key3'], reg.get('key3'))
        eq_(NO_VALUE, reg.get('key10'))

    def test_region_set_get_nothing(self):
        reg = self._region()
        some_key = gen_some_key()
        reg.delete_multi([some_key])
        eq_(reg.get(some_key), NO_VALUE)

    def test_region_creator(self):
        reg = self._region()

        def creator():
            return 'some value'
        some_key = gen_some_key()
        eq_(reg.get_or_create(some_key, creator), 'some value')

    @pytest.mark.time_intensive
    def test_threaded_dogpile(self):
        reg = self._region(config_args={'expiration_time': 0.25})
        lock = Lock()
        canary = []
        some_key = gen_some_key()

        def creator():
            ack = lock.acquire(False)
            canary.append(ack)
            time.sleep(0.25)
            if ack:
                lock.release()
            return 'some value'

        def f():
            for x in range(5):
                reg.get_or_create(some_key, creator)
                time.sleep(0.5)
        threads = [Thread(target=f) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(canary) > 2
        if not reg.backend.has_lock_timeout():
            assert False not in canary

    @pytest.mark.time_intensive
    def test_threaded_get_multi(self):
        """This test is testing that when we get inside the "creator" for
        a certain key, there are no other "creators" running at all for
        that key.

        With "distributed" locks, this is not 100% the case.

        """
        some_key = gen_some_key()
        reg = self._region(config_args={'expiration_time': 0.25})
        backend_mutex = reg.backend.get_mutex(some_key)
        is_custom_mutex = backend_mutex is not None
        locks = dict(((str(i), Lock()) for i in range(11)))
        canary = collections.defaultdict(list)

        def creator(*keys):
            assert keys
            ack = [locks[key].acquire(False) for key in keys]
            for acq, key in zip(ack, keys):
                canary[key].append(acq)
            time.sleep(0.5)
            for acq, key in zip(ack, keys):
                if acq:
                    locks[key].release()
            return ['some value %s' % k for k in keys]

        def f():
            for x in range(5):
                reg.get_or_create_multi([str(random.randint(1, 10)) for i in range(random.randint(1, 5))], creator)
                time.sleep(0.5)
        f()
        threads = [Thread(target=f) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sum([len(v) for v in canary.values()]) > 10
        if not is_custom_mutex:
            for l in canary.values():
                assert False not in l

    def test_region_delete(self):
        reg = self._region()
        some_key = gen_some_key()
        reg.set(some_key, 'some value')
        reg.delete(some_key)
        reg.delete(some_key)
        eq_(reg.get(some_key), NO_VALUE)

    @pytest.mark.time_intensive
    def test_region_expire(self):
        some_key = gen_some_key()
        expire_time = 1.0
        reg = self._region(config_args={'expiration_time': expire_time})
        counter = itertools.count(1)

        def creator():
            return 'some value %d' % next(counter)
        eq_(reg.get_or_create(some_key, creator), 'some value 1')
        time.sleep(expire_time + 0.2 * expire_time)
        post_expiration = reg.get(some_key, ignore_expiration=True)
        if post_expiration is not NO_VALUE:
            eq_(post_expiration, 'some value 1')
        eq_(reg.get_or_create(some_key, creator), 'some value 2')
        eq_(reg.get(some_key), 'some value 2')

    def test_decorated_fn_functionality(self):
        reg = self._region()
        counter = itertools.count(1)

        @reg.cache_on_arguments()
        def my_function(x, y):
            return next(counter) + x + y
        my_function.invalidate(3, 4)
        my_function.invalidate(5, 6)
        my_function.invalidate(4, 3)
        eq_(my_function(3, 4), 8)
        eq_(my_function(5, 6), 13)
        eq_(my_function(3, 4), 8)
        eq_(my_function(4, 3), 10)
        my_function.invalidate(4, 3)
        eq_(my_function(4, 3), 11)

    def test_exploding_value_fn(self):
        some_key = gen_some_key()
        reg = self._region()

        def boom():
            raise Exception('boom')
        assert_raises_message(Exception, 'boom', reg.get_or_create, some_key, boom)