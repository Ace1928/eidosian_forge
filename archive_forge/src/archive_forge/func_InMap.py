import array
import contextlib
import enum
import struct
def InMap(func):

    def wrapper(self, *args, **kwargs):
        if isinstance(args[0], str):
            self.Key(args[0])
            func(self, *args[1:], **kwargs)
        else:
            func(self, *args, **kwargs)
    return wrapper