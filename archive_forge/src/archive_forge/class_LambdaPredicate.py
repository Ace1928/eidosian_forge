import contextlib
import operator
import re
import sys
from . import config
from .. import util
from ..util import decorator
from ..util.compat import inspect_getfullargspec
class LambdaPredicate(Predicate):

    def __init__(self, lambda_, description=None, args=None, kw=None):
        spec = inspect_getfullargspec(lambda_)
        if not spec[0]:
            self.lambda_ = lambda db: lambda_()
        else:
            self.lambda_ = lambda_
        self.args = args or ()
        self.kw = kw or {}
        if description:
            self.description = description
        elif lambda_.__doc__:
            self.description = lambda_.__doc__
        else:
            self.description = 'custom function'

    def __call__(self, config):
        return self.lambda_(config)

    def _as_string(self, config, negate=False):
        return self._format_description(config)