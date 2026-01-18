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