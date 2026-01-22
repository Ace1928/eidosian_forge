import sys
from past.utils import with_metaclass, PY2
class BaseBaseString(type):

    def __instancecheck__(cls, instance):
        return isinstance(instance, (bytes, str))

    def __subclasscheck__(cls, subclass):
        return super(BaseBaseString, cls).__subclasscheck__(subclass) or issubclass(subclass, (bytes, str))