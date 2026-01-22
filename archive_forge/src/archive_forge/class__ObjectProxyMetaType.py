import sys
import operator
import inspect
class _ObjectProxyMetaType(type):

    def __new__(cls, name, bases, dictionary):
        dictionary.update(vars(_ObjectProxyMethods))
        return type.__new__(cls, name, bases, dictionary)