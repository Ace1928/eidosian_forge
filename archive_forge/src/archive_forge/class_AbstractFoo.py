import abc
import unittest
import warnings
from traits.api import ABCHasTraits, ABCMetaHasTraits, HasTraits, Int, Float
class AbstractFoo(ABCHasTraits):
    x = Int(10)
    y = Float(20.0)

    @abc.abstractmethod
    def foo(self):
        raise NotImplementedError()

    @abc.abstractproperty
    def bar(self):
        raise NotImplementedError()