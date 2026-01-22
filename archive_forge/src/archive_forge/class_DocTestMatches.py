import doctest
import re
from ._impl import Mismatch
class DocTestMatches:
    """See if a string matches a doctest example."""

    def __init__(self, example, flags=0):
        """Create a DocTestMatches to match example.

        :param example: The example to match e.g. 'foo bar baz'
        :param flags: doctest comparison flags to match on. e.g.
            doctest.ELLIPSIS.
        """
        if not example.endswith('\n'):
            example += '\n'
        self.want = example
        self.flags = flags
        self._checker = _NonManglingOutputChecker()

    def __str__(self):
        if self.flags:
            flagstr = ', flags=%d' % self.flags
        else:
            flagstr = ''
        return f'DocTestMatches({self.want!r}{flagstr})'

    def _with_nl(self, actual):
        result = self.want.__class__(actual)
        if not result.endswith('\n'):
            result += '\n'
        return result

    def match(self, actual):
        with_nl = self._with_nl(actual)
        if self._checker.check_output(self.want, with_nl, self.flags):
            return None
        return DocTestMismatch(self, with_nl)

    def _describe_difference(self, with_nl):
        return self._checker.output_difference(self, with_nl, self.flags)