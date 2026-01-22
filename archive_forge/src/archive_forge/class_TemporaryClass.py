import collections
import contextlib
import errno
import functools
import os
import sys
import types
class TemporaryClass(base_exception):

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], TemporaryClass):
            unwrap_me = args[0]
            for attr in dir(unwrap_me):
                if not attr.startswith('__'):
                    setattr(self, attr, getattr(unwrap_me, attr))
        else:
            super(TemporaryClass, self).__init__(*args, **kwargs)

    class __metaclass__(type):

        def __instancecheck__(cls, inst):
            return instance_checker(inst)

        def __subclasscheck__(cls, classinfo):
            value = sys.exc_info()[1]
            return isinstance(value, cls)