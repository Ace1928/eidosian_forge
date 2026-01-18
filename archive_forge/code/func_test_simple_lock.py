import doctest
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from zope.testing import setupstack
import zc.lockfile
def test_simple_lock(self):
    assert isinstance(zc.lockfile.SimpleLockFile, type)
    lock = zc.lockfile.SimpleLockFile('s')
    with self.assertRaises(zc.lockfile.LockError):
        zc.lockfile.SimpleLockFile('s')
    lock.close()
    zc.lockfile.SimpleLockFile('s').close()