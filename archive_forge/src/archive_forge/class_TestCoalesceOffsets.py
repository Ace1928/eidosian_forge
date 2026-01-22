import errno
import os
import subprocess
import sys
import threading
from io import BytesIO
import breezy.transport.trace
from .. import errors, osutils, tests, transport, urlutils
from ..transport import (FileExists, NoSuchFile, UnsupportedProtocol, chroot,
from . import features, test_server
class TestCoalesceOffsets(tests.TestCase):

    def check(self, expected, offsets, limit=0, max_size=0, fudge=0):
        coalesce = transport.Transport._coalesce_offsets
        exp = [transport._CoalescedOffset(*x) for x in expected]
        out = list(coalesce(offsets, limit=limit, fudge_factor=fudge, max_size=max_size))
        self.assertEqual(exp, out)

    def test_coalesce_empty(self):
        self.check([], [])

    def test_coalesce_simple(self):
        self.check([(0, 10, [(0, 10)])], [(0, 10)])

    def test_coalesce_unrelated(self):
        self.check([(0, 10, [(0, 10)]), (20, 10, [(0, 10)])], [(0, 10), (20, 10)])

    def test_coalesce_unsorted(self):
        self.check([(20, 10, [(0, 10)]), (0, 10, [(0, 10)])], [(20, 10), (0, 10)])

    def test_coalesce_nearby(self):
        self.check([(0, 20, [(0, 10), (10, 10)])], [(0, 10), (10, 10)])

    def test_coalesce_overlapped(self):
        self.assertRaises(ValueError, self.check, [(0, 15, [(0, 10), (5, 10)])], [(0, 10), (5, 10)])

    def test_coalesce_limit(self):
        self.check([(10, 50, [(0, 10), (10, 10), (20, 10), (30, 10), (40, 10)]), (60, 50, [(0, 10), (10, 10), (20, 10), (30, 10), (40, 10)])], [(10, 10), (20, 10), (30, 10), (40, 10), (50, 10), (60, 10), (70, 10), (80, 10), (90, 10), (100, 10)], limit=5)

    def test_coalesce_no_limit(self):
        self.check([(10, 100, [(0, 10), (10, 10), (20, 10), (30, 10), (40, 10), (50, 10), (60, 10), (70, 10), (80, 10), (90, 10)])], [(10, 10), (20, 10), (30, 10), (40, 10), (50, 10), (60, 10), (70, 10), (80, 10), (90, 10), (100, 10)])

    def test_coalesce_fudge(self):
        self.check([(10, 30, [(0, 10), (20, 10)]), (100, 10, [(0, 10)])], [(10, 10), (30, 10), (100, 10)], fudge=10)

    def test_coalesce_max_size(self):
        self.check([(10, 20, [(0, 10), (10, 10)]), (30, 50, [(0, 50)]), (100, 80, [(0, 80)])], [(10, 10), (20, 10), (30, 50), (100, 80)], max_size=50)

    def test_coalesce_no_max_size(self):
        self.check([(10, 170, [(0, 10), (10, 10), (20, 50), (70, 100)])], [(10, 10), (20, 10), (30, 50), (80, 100)])

    def test_coalesce_default_limit(self):
        ten_mb = 10 * 1024 * 1024
        self.check([(0, 10 * ten_mb, [(i * ten_mb, ten_mb) for i in range(10)]), (10 * ten_mb, ten_mb, [(0, ten_mb)])], [(i * ten_mb, ten_mb) for i in range(11)])
        self.check([(0, 11 * ten_mb, [(i * ten_mb, ten_mb) for i in range(11)])], [(i * ten_mb, ten_mb) for i in range(11)], max_size=1 * 1024 * 1024 * 1024)