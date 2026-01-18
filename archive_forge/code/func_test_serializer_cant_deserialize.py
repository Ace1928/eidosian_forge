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
def test_serializer_cant_deserialize(self):
    region = self._region(region_args={'serializer': self.region_args['serializer'], 'deserializer': raise_cant_deserialize_exception})
    value = {'foo': ['bar', 1, False, None]}
    region.set('k', value)
    asserted = region.get('k')
    eq_(asserted, NO_VALUE)