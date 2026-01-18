import operator
import sys
import types
import unittest
import abc
import pytest
import six
@pytest.mark.skipif('sys.version_info[:2] < (3, 7)')
def test_with_metaclass_pep_560():

    class Meta(type):
        pass

    class A:
        pass

    class B:
        pass

    class Fake:

        def __mro_entries__(self, bases):
            return (A, B)
    fake = Fake()

    class G(six.with_metaclass(Meta, fake)):
        pass

    class GA(six.with_metaclass(abc.ABCMeta, fake)):
        pass
    assert isinstance(G, Meta)
    assert isinstance(GA, abc.ABCMeta)
    assert G.__bases__ == (A, B)
    assert G.__orig_bases__ == (fake,)