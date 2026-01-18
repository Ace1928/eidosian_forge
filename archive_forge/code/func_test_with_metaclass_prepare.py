import operator
import sys
import types
import unittest
import abc
import pytest
import six
@pytest.mark.skipif('sys.version_info[:2] < (3, 0)')
def test_with_metaclass_prepare():
    """Test that with_metaclass causes Meta.__prepare__ to be called with the correct arguments."""

    class MyDict(dict):
        pass

    class Meta(type):

        @classmethod
        def __prepare__(cls, name, bases):
            namespace = MyDict(super().__prepare__(name, bases), cls=cls, bases=bases)
            namespace['namespace'] = namespace
            return namespace

    class Base(object):
        pass
    bases = (Base,)

    class X(six.with_metaclass(Meta, *bases)):
        pass
    assert getattr(X, 'cls', type) is Meta
    assert getattr(X, 'bases', ()) == bases
    assert isinstance(getattr(X, 'namespace', {}), MyDict)