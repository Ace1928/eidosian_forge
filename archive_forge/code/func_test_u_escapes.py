import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_u_escapes():
    s = six.u('ሴ')
    assert len(s) == 1