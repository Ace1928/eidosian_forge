from __future__ import absolute_import
import os
from .. import Utils
class ShouldBeFromDirective(object):
    known_directives = []

    def __init__(self, options_name, directive_name=None, disallow=False):
        self.options_name = options_name
        self.directive_name = directive_name or options_name
        self.disallow = disallow
        self.known_directives.append(self)

    def __nonzero__(self):
        self._bad_access()

    def __int__(self):
        self._bad_access()

    def _bad_access(self):
        raise RuntimeError(repr(self))

    def __repr__(self):
        return "Illegal access of '%s' from Options module rather than directive '%s'" % (self.options_name, self.directive_name)