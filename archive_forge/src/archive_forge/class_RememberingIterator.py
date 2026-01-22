import collections
import re
import sys
from yaql.language import exceptions
from yaql.language import lexer
class RememberingIterator:

    def __init__(self):
        self.seq = iter(collection)
        self.index = 0

    def __iter__(self):
        return RememberingIterator()

    def __next__(self):
        if self.index < len(yielded):
            self.index += 1
            return yielded[self.index - 1]
        else:
            val = next(self.seq)
            yielded.append(val)
            limit_memory_usage(engine, (1, yielded))
            self.index += 1
            return val