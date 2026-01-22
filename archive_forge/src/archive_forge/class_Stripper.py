import re
import itertools
import textwrap
import functools
from jaraco.functools import compose, method_cache
from jaraco.context import ExceptionTrap
class Stripper:
    """
    Given a series of lines, find the common prefix and strip it from them.

    >>> lines = [
    ...     'abcdefg\\n',
    ...     'abc\\n',
    ...     'abcde\\n',
    ... ]
    >>> res = Stripper.strip_prefix(lines)
    >>> res.prefix
    'abc'
    >>> list(res.lines)
    ['defg\\n', '\\n', 'de\\n']

    If no prefix is common, nothing should be stripped.

    >>> lines = [
    ...     'abcd\\n',
    ...     '1234\\n',
    ... ]
    >>> res = Stripper.strip_prefix(lines)
    >>> res.prefix = ''
    >>> list(res.lines)
    ['abcd\\n', '1234\\n']
    """

    def __init__(self, prefix, lines):
        self.prefix = prefix
        self.lines = map(self, lines)

    @classmethod
    def strip_prefix(cls, lines):
        prefix_lines, lines = itertools.tee(lines)
        prefix = functools.reduce(cls.common_prefix, prefix_lines)
        return cls(prefix, lines)

    def __call__(self, line):
        if not self.prefix:
            return line
        null, prefix, rest = line.partition(self.prefix)
        return rest

    @staticmethod
    def common_prefix(s1, s2):
        """
        Return the common prefix of two lines.
        """
        index = min(len(s1), len(s2))
        while s1[:index] != s2[:index]:
            index -= 1
        return s1[:index]