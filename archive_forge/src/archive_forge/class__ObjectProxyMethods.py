import sys
import operator
import inspect
class _ObjectProxyMethods(object):

    @property
    def __module__(self):
        return self.__wrapped__.__module__

    @__module__.setter
    def __module__(self, value):
        self.__wrapped__.__module__ = value

    @property
    def __doc__(self):
        return self.__wrapped__.__doc__

    @__doc__.setter
    def __doc__(self, value):
        self.__wrapped__.__doc__ = value

    @property
    def __dict__(self):
        return self.__wrapped__.__dict__

    @property
    def __weakref__(self):
        return self.__wrapped__.__weakref__