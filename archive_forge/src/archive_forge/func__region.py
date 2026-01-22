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
def _region(self, backend=None, region_args={}, config_args={}):
    _region_args = {}
    for cls in reversed(self.__class__.__mro__):
        if 'region_args' in cls.__dict__:
            _region_args.update(cls.__dict__['region_args'])
    _region_args.update(**region_args)
    _config_args = self.config_args.copy()
    _config_args.update(config_args)

    def _store_keys(key):
        if existing_key_mangler:
            key = existing_key_mangler(key)
        self._keys.add(key)
        return key
    self._region_inst = reg = CacheRegion(**_region_args)
    existing_key_mangler = self._region_inst.key_mangler
    self._region_inst.key_mangler = _store_keys
    self._region_inst._user_defined_key_mangler = _store_keys
    reg.configure(backend or self.backend, **_config_args)
    return reg