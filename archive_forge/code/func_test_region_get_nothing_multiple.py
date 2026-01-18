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
def test_region_get_nothing_multiple(self):
    reg = self._region()
    reg.delete_multi(['key1', 'key2', 'key3', 'key4', 'key5'])
    values = {'key1': 'value1', 'key3': 'value3', 'key5': 'value5'}
    reg.set_multi(values)
    reg_values = reg.get_multi(['key1', 'key2', 'key3', 'key4', 'key5', 'key6'])
    eq_(reg_values, ['value1', NO_VALUE, 'value3', NO_VALUE, 'value5', NO_VALUE])