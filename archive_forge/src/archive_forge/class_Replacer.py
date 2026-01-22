import re
from . import lazy_regex
from .trace import mutter, warning
class Replacer:
    """Do a multiple-pattern substitution.

    The patterns and substitutions are combined into one, so the result of
    one replacement is never substituted again. Add the patterns and
    replacements via the add method and then call the object. The patterns
    must not contain capturing groups.
    """
    _expand = lazy_regex.lazy_compile('\\\\&')

    def __init__(self, source=None):
        self._pat = None
        if source:
            self._pats = list(source._pats)
            self._funs = list(source._funs)
        else:
            self._pats = []
            self._funs = []

    def add(self, pat, fun):
        """Add a pattern and replacement.

        The pattern must not contain capturing groups.
        The replacement might be either a string template in which \\& will be
        replaced with the match, or a function that will get the matching text
        as argument. It does not get match object, because capturing is
        forbidden anyway.
        """
        self._pat = None
        self._pats.append(pat)
        self._funs.append(fun)

    def add_replacer(self, replacer):
        """Add all patterns from another replacer.

        All patterns and replacements from replacer are appended to the ones
        already defined.
        """
        self._pat = None
        self._pats.extend(replacer._pats)
        self._funs.extend(replacer._funs)

    def __call__(self, text):
        if not self._pat:
            self._pat = lazy_regex.lazy_compile('|'.join(['(%s)' % p for p in self._pats]), re.UNICODE)
        return self._pat.sub(self._do_sub, text)

    def _do_sub(self, m):
        fun = self._funs[m.lastindex - 1]
        if hasattr(fun, '__call__'):
            return fun(m.group(0))
        else:
            return self._expand.sub(m.group(0), fun)