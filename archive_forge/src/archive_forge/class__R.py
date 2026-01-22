import logging
import re
class _R:

    def __init__(self, i):
        self.i = i

    def __call__(self):
        v = self.i
        self.i += 1
        return v

    def value(self):
        return self.i