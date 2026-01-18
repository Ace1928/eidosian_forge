import pytest
import tempfile
import os
from unittest.mock import patch
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.resources import resource_find, resource_add_path
def test_timestamp_and_lastaccess(test_file):
    Cache.remove(RESOURCE_CACHE)
    start = Clock.get_time()
    resource_find(test_file)
    ts = Cache.get_timestamp(RESOURCE_CACHE, test_file)
    last_access = Cache.get_lastaccess(RESOURCE_CACHE, test_file)
    assert ts >= start, 'Last timestamp not accurate.'
    assert last_access >= start, 'Last access time is not accurate.'