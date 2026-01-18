import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_wraps_raises_on_missing_updated_field_on_wrapper():
    """Ensure six.wraps doesn't ignore missing attrs wrapper.

    Because that's what happens in Py3's functools.update_wrapper.
    """

    def wrapped():
        pass

    def wrapper():
        pass
    with pytest.raises(AttributeError, match='has no attribute.*xyzzy'):
        six.wraps(wrapped, [], ['xyzzy'])(wrapper)