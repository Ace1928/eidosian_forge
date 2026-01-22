from collections import defaultdict, deque
import itertools
import pprint
import textwrap
from jsonschema import _utils
from jsonschema.compat import PY3, iteritems
def _contents(self):
    attrs = ('message', 'cause', 'context', 'validator', 'validator_value', 'path', 'schema_path', 'instance', 'schema', 'parent')
    return dict(((attr, getattr(self, attr)) for attr in attrs))