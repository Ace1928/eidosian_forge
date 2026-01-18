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
def test_region_set_multiple_values(self):
    reg = self._region()
    values = {'key1': 'value1', 'key2': 'value2', 'key3': 'value3'}
    reg.set_multi(values)
    eq_(values['key1'], reg.get('key1'))
    eq_(values['key2'], reg.get('key2'))
    eq_(values['key3'], reg.get('key3'))