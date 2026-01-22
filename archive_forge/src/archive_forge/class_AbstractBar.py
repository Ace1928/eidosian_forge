import abc
import unittest
import warnings
from traits.api import ABCHasTraits, ABCMetaHasTraits, HasTraits, Int, Float
class AbstractBar(abc.ABC):
    pass

    @abc.abstractmethod
    def bar(self):
        raise NotImplementedError()