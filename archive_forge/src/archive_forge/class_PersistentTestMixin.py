import os
import glob
from abc import abstractmethod
from unittest import mock
from typing import Any, Optional
import pytest
from queuelib.queue import (
from queuelib.tests import QueuelibTestCase
class PersistentTestMixin:
    chunksize = 100000

    @pytest.mark.xfail(reason='Reenable once Scrapy.squeues stops extending from this testsuite')
    def test_non_bytes_raises_typeerror(self):
        q = self.queue()
        self.assertRaises(TypeError, q.push, 0)
        self.assertRaises(TypeError, q.push, u'')
        self.assertRaises(TypeError, q.push, None)
        self.assertRaises(TypeError, q.push, lambda x: x)

    def test_text_in_windows(self):
        e1 = b'\r\n'
        q = self.queue()
        q.push(e1)
        q.close()
        q = self.queue()
        e2 = q.pop()
        self.assertEqual(e1, e2)

    def test_close_open(self):
        """Test closing and re-opening keeps state"""
        q = self.queue()
        q.push(b'a')
        q.push(b'b')
        q.push(b'c')
        q.push(b'd')
        q.pop()
        q.pop()
        q.close()
        del q
        q = self.queue()
        self.assertEqual(len(q), 2)
        q.push(b'e')
        q.pop()
        q.pop()
        q.close()
        del q
        q = self.queue()
        assert q.pop() is not None
        self.assertEqual(len(q), 0)

    def test_cleanup(self):
        """Test queue dir is removed if queue is empty"""
        q = self.queue()
        values = [b'0', b'1', b'2', b'3', b'4']
        assert os.path.exists(self.qpath)
        for x in values:
            q.push(x)
        for x in values:
            q.pop()
        q.close()
        assert not os.path.exists(self.qpath)