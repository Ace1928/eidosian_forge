from collections import defaultdict, OrderedDict
from contextlib import contextmanager
import sys
import gast as ast
@contextmanager
def CompDefinitionContext(self, node):
    if sys.version_info.major >= 3:
        self._currenthead.append(node)
        self._definitions.append(defaultdict(ordered_set))
        self._promoted_locals.append(set())
    yield
    if sys.version_info.major >= 3:
        self._promoted_locals.pop()
        self._definitions.pop()
        self._currenthead.pop()