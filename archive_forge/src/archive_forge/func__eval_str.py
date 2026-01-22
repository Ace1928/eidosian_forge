import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def _eval_str(self, config, negate=False):
    if negate:
        conjunction = ' and '
    else:
        conjunction = ' or '
    return conjunction.join((p._as_string(config, negate=negate) for p in self.predicates))