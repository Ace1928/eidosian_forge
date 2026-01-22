import sys
import copy
from future.utils import with_metaclass
from future.types.newobject import newobject
class BaseNewList(type):

    def __instancecheck__(cls, instance):
        if cls == newlist:
            return isinstance(instance, _builtin_list)
        else:
            return issubclass(instance.__class__, cls)