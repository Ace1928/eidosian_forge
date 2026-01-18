import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_zip_longest():
    from six.moves import zip_longest
    it = zip_longest(range(2), range(1))
    assert six.advance_iterator(it) == (0, 0)
    assert six.advance_iterator(it) == (1, None)