import functools
import re
import os
class PushBackIterator:

    def __init__(self, iterator):
        self.pushes = []
        self.iterator = iterator
        self.current = None

    def push_back(self, value):
        self.pushes.append(value)

    def __iter__(self):
        return self

    def __next__(self):
        if self.pushes:
            self.current = self.pushes.pop()
        else:
            self.current = next(self.iterator)
        return self.current