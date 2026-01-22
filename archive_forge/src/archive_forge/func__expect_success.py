import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
def _expect_success(self, config, name='block'):
    if not self.fails:
        return
    for fail in self.fails:
        if fail(config):
            raise AssertionError("Unexpected success for '%s' (%s)" % (name, ' and '.join((fail._as_string(config) for fail in self.fails))))