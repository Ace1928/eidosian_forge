import io as abc
from zope.interface.common import ABCInterface
class IBufferedIOBase(IIOBase):
    abc = abc.BufferedIOBase
    extra_classes = ()