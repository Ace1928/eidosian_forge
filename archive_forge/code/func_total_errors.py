from collections import defaultdict, deque
import itertools
import pprint
import textwrap
from jsonschema import _utils
from jsonschema.compat import PY3, iteritems
@property
def total_errors(self):
    """
        The total number of errors in the entire tree, including children.

        """
    child_errors = sum((len(tree) for _, tree in iteritems(self._contents)))
    return len(self.errors) + child_errors