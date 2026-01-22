import ast
import collections
import errno
import json
import os
import pickle
import socket
import stat
import unittest
import psutil
import psutil.tests
from psutil import LINUX
from psutil import POSIX
from psutil import WINDOWS
from psutil._common import bcat
from psutil._common import cat
from psutil._common import debug
from psutil._common import isfile_strict
from psutil._common import memoize
from psutil._common import memoize_when_activated
from psutil._common import parse_environ_block
from psutil._common import supports_ipv6
from psutil._common import wrap_numbers
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import redirect_stderr
from psutil.tests import APPVEYOR
from psutil.tests import CI_TESTING
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import SCRIPTS_DIR
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import reload_module
from psutil.tests import sh
class TestWrapNumbers(PsutilTestCase):

    def setUp(self):
        wrap_numbers.cache_clear()
    tearDown = setUp

    def test_first_call(self):
        input = {'disk1': nt(5, 5, 5)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)

    def test_input_hasnt_changed(self):
        input = {'disk1': nt(5, 5, 5)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)

    def test_increase_but_no_wrap(self):
        input = {'disk1': nt(5, 5, 5)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)
        input = {'disk1': nt(10, 15, 20)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)
        input = {'disk1': nt(20, 25, 30)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)
        input = {'disk1': nt(20, 25, 30)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)

    def test_wrap(self):
        input = {'disk1': nt(100, 100, 100)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)
        input = {'disk1': nt(100, 100, 10)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), {'disk1': nt(100, 100, 110)})
        input = {'disk1': nt(100, 100, 10)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), {'disk1': nt(100, 100, 110)})
        input = {'disk1': nt(100, 100, 90)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), {'disk1': nt(100, 100, 190)})
        input = {'disk1': nt(100, 100, 20)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), {'disk1': nt(100, 100, 210)})
        input = {'disk1': nt(100, 100, 20)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), {'disk1': nt(100, 100, 210)})
        input = {'disk1': nt(50, 100, 20)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), {'disk1': nt(150, 100, 210)})
        input = {'disk1': nt(40, 100, 20)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), {'disk1': nt(190, 100, 210)})
        input = {'disk1': nt(40, 100, 20)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), {'disk1': nt(190, 100, 210)})

    def test_changing_keys(self):
        input = {'disk1': nt(5, 5, 5)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)
        input = {'disk1': nt(5, 5, 5), 'disk2': nt(7, 7, 7)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)
        input = {'disk1': nt(8, 8, 8)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)

    def test_changing_keys_w_wrap(self):
        input = {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 100)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)
        input = {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 10)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 110)})
        input = {'disk1': nt(50, 50, 50)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)
        input = {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 100)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)
        input = {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 100)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), input)
        input = {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 10)}
        self.assertEqual(wrap_numbers(input, 'disk_io'), {'disk1': nt(50, 50, 50), 'disk2': nt(100, 100, 110)})

    def test_real_data(self):
        d = {'nvme0n1': (300, 508, 640, 1571, 5970, 1987, 2049, 451751, 47048), 'nvme0n1p1': (1171, 2, 5600256, 1024, 516, 0, 0, 0, 8), 'nvme0n1p2': (54, 54, 2396160, 5165056, 4, 24, 30, 1207, 28), 'nvme0n1p3': (2389, 4539, 5154, 150, 4828, 1844, 2019, 398, 348)}
        self.assertEqual(wrap_numbers(d, 'disk_io'), d)
        self.assertEqual(wrap_numbers(d, 'disk_io'), d)
        d = {'nvme0n1': (100, 508, 640, 1571, 5970, 1987, 2049, 451751, 47048), 'nvme0n1p1': (1171, 2, 5600256, 1024, 516, 0, 0, 0, 8), 'nvme0n1p2': (54, 54, 2396160, 5165056, 4, 24, 30, 1207, 28), 'nvme0n1p3': (2389, 4539, 5154, 150, 4828, 1844, 2019, 398, 348)}
        out = wrap_numbers(d, 'disk_io')
        self.assertEqual(out['nvme0n1'][0], 400)

    def test_cache_first_call(self):
        input = {'disk1': nt(5, 5, 5)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        self.assertEqual(cache[0], {'disk_io': input})
        self.assertEqual(cache[1], {'disk_io': {}})
        self.assertEqual(cache[2], {'disk_io': {}})

    def test_cache_call_twice(self):
        input = {'disk1': nt(5, 5, 5)}
        wrap_numbers(input, 'disk_io')
        input = {'disk1': nt(10, 10, 10)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        self.assertEqual(cache[0], {'disk_io': input})
        self.assertEqual(cache[1], {'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 0}})
        self.assertEqual(cache[2], {'disk_io': {}})

    def test_cache_wrap(self):
        input = {'disk1': nt(100, 100, 100)}
        wrap_numbers(input, 'disk_io')
        input = {'disk1': nt(100, 100, 10)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        self.assertEqual(cache[0], {'disk_io': input})
        self.assertEqual(cache[1], {'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 100}})
        self.assertEqual(cache[2], {'disk_io': {'disk1': set([('disk1', 2)])}})

        def check_cache_info():
            cache = wrap_numbers.cache_info()
            self.assertEqual(cache[1], {'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 100}})
            self.assertEqual(cache[2], {'disk_io': {'disk1': set([('disk1', 2)])}})
        input = {'disk1': nt(100, 100, 10)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        self.assertEqual(cache[0], {'disk_io': input})
        check_cache_info()
        input = {'disk1': nt(100, 100, 90)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        self.assertEqual(cache[0], {'disk_io': input})
        check_cache_info()
        input = {'disk1': nt(100, 100, 20)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        self.assertEqual(cache[0], {'disk_io': input})
        self.assertEqual(cache[1], {'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 190}})
        self.assertEqual(cache[2], {'disk_io': {'disk1': set([('disk1', 2)])}})

    def test_cache_changing_keys(self):
        input = {'disk1': nt(5, 5, 5)}
        wrap_numbers(input, 'disk_io')
        input = {'disk1': nt(5, 5, 5), 'disk2': nt(7, 7, 7)}
        wrap_numbers(input, 'disk_io')
        cache = wrap_numbers.cache_info()
        self.assertEqual(cache[0], {'disk_io': input})
        self.assertEqual(cache[1], {'disk_io': {('disk1', 0): 0, ('disk1', 1): 0, ('disk1', 2): 0}})
        self.assertEqual(cache[2], {'disk_io': {}})

    def test_cache_clear(self):
        input = {'disk1': nt(5, 5, 5)}
        wrap_numbers(input, 'disk_io')
        wrap_numbers(input, 'disk_io')
        wrap_numbers.cache_clear('disk_io')
        self.assertEqual(wrap_numbers.cache_info(), ({}, {}, {}))
        wrap_numbers.cache_clear('disk_io')
        wrap_numbers.cache_clear('?!?')

    @unittest.skipIf(not HAS_NET_IO_COUNTERS, 'not supported')
    def test_cache_clear_public_apis(self):
        if not psutil.disk_io_counters() or not psutil.net_io_counters():
            return self.skipTest('no disks or NICs available')
        psutil.disk_io_counters()
        psutil.net_io_counters()
        caches = wrap_numbers.cache_info()
        for cache in caches:
            self.assertIn('psutil.disk_io_counters', cache)
            self.assertIn('psutil.net_io_counters', cache)
        psutil.disk_io_counters.cache_clear()
        caches = wrap_numbers.cache_info()
        for cache in caches:
            self.assertIn('psutil.net_io_counters', cache)
            self.assertNotIn('psutil.disk_io_counters', cache)
        psutil.net_io_counters.cache_clear()
        caches = wrap_numbers.cache_info()
        self.assertEqual(caches, ({}, {}, {}))