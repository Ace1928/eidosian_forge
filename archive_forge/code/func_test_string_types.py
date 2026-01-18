import operator
import sys
import types
import unittest
import abc
import pytest
import six
def test_string_types():
    assert isinstance('hi', six.string_types)
    assert isinstance(six.u('hi'), six.string_types)
    assert issubclass(six.text_type, six.string_types)