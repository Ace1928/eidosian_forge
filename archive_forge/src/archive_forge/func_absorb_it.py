import collections
from collections import abc
import itertools
def absorb_it(sets):
    for value in iter(self):
        seen = False
        for s in sets:
            if value in s:
                seen = True
                break
        if not seen:
            yield value