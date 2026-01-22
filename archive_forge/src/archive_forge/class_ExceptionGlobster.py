import re
from . import lazy_regex
from .trace import mutter, warning
class ExceptionGlobster:
    """A Globster that supports exception patterns.

    Exceptions are ignore patterns prefixed with '!'.  Exception
    patterns take precedence over regular patterns and cause a
    matching filename to return None from the match() function.
    Patterns using a '!!' prefix are highest precedence, and act
    as regular ignores. '!!' patterns are useful to establish ignores
    that apply under paths specified by '!' exception patterns.
    """

    def __init__(self, patterns):
        ignores = [[], [], []]
        for p in patterns:
            if p.startswith('!!'):
                ignores[2].append(p[2:])
            elif p.startswith('!'):
                ignores[1].append(p[1:])
            else:
                ignores[0].append(p)
        self._ignores = [Globster(i) for i in ignores]

    def match(self, filename):
        """Searches for a pattern that matches the given filename.

        :return A matching pattern or None if there is no matching pattern.
        """
        double_neg = self._ignores[2].match(filename)
        if double_neg:
            return '!!%s' % double_neg
        elif self._ignores[1].match(filename):
            return None
        else:
            return self._ignores[0].match(filename)