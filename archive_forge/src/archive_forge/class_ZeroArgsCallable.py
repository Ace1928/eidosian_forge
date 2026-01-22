import inspect
import unittest
from traits.api import (
class ZeroArgsCallable(BaseCallable):

    def validate(self, object, name, value):
        if callable(value):
            sig = inspect.signature(value)
            if len(sig.parameters) == 0:
                return value
        self.error(object, name, value)