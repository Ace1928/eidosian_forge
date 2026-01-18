# Licensed under the GPL: https://www.gnu.org/licenses/gpl-3.0.en.html
# For details: reprotest/debian/copyright

import collections
import itertools


### Formatting

class _Tuple(tuple):
    '''Tuple subclass that returns appropriate types from methods.

    This overloads tuple methods so they return the subclass's type
    rather than tuple and provides a nicer __repr__.
    '''
    __slots__ = ()

    def __add__(self, other):
        if self.__class__ is other.__class__:
            return self.__class__(itertools.chain(self, other))
        else:
            raise TypeError('Cannot add two shell AST nodes of different types.')
    __iadd__ = __add__

    def __radd__(self, other):
        if self.__class__ is other.__class__:
            return self.__class__(itertools.chain(other, self))
        else:
            raise TypeError('Cannot add two shell AST nodes of different types: %s, %s' % (repr(self), repr(other)))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return self.__class__(super().__getitem__(index))
        else:
            return super().__getitem__(index)

    def __repr__(self):
        return self.__class__.__name__ + super().__repr__()


    def __str__(self, indent=None):
        return ' '.join(str(field) for field in self)


class CmdPrefix(_Tuple):
    pass

class CmdSuffix(_Tuple):
    pass

class Command(collections.namedtuple('_Command', 'cmd_prefix cmd_suffix')):
    '''A command arbitrarily divided as (prefix, suffix) for formatting.'''
    def __str__(self, indent=None):
        sep = ' ' if indent is None else ' \\\n%s' % (" " * indent)
        return ((self.cmd_prefix.__str__(indent) + sep if self.cmd_prefix else '') +
                (self.cmd_suffix.__str__(indent) if self.cmd_suffix else ''))

    @classmethod
    def make(cls, *args):
        return cls(CmdPrefix(), CmdSuffix(args))


class List(_Tuple):
    '''List of commands separated by semicolon, for formatting.'''
    def __str__(self, indent=None):
        sep = '; ' if indent is None else '; \\\n%s' % (" " * indent)
        return sep.join(field.__str__(indent) for field in self)

    @classmethod
    def make(cls, *args):
        return cls(list(map(Command.make, args)))


class AndList(_Tuple):
    '''List of commands separated by &&, for formatting.'''
    def __str__(self, indent=None):
        sep = ' && ' if indent is None else ' && \\\n%s' % (" " * indent)
        return sep.join(field.__str__(indent) for field in self)


### Parsing

# http://pubs.opengroup.org/onlinepubs/9699919799/utilities/V3_chap02.html
# Special chars except allowed ones <space> " ' \
special_except_quotes=r"""
	|&;<>()$`"""
special_in_double_quotes=r"$`"
escaped_in_double_quotes=r"""$`"\
"""

def sanitize_globs(s, join=True, forcerel=True):
    """Ensure that a shell snippet only contains glob expressions, and nothing
    that can be executed.

    Args:

    s: The snippet to sanitize
    join: If true, returns a single shell snippet string, else a list of shell
        snippet words (that include quotes and other limit) that may be joined
        by a single space into a single shell snippet. Default: true.
    forcerel: If true, prepends ./ to all snippet words. This ensures that they
        can't be accidentally interpreted as options to most commands.
    """
    # output state
    words = []
    cw = None
    def next_word():
        nonlocal cw, words
        if cw is not None:
            words.append(cw)
            cw = None

    # parse state
    in_quote = False
    escaped = False

    for c in s:
        #print(c, in_quote, escaped)
        if not in_quote:
            if escaped:
                escaped = False
            else:
                if c in special_except_quotes:
                    raise ValueError("not a shell-glob pattern: %s" % s)
                elif c == "\\":
                    escaped = True
                elif c in "'\"":
                    in_quote = c
                elif c == " ":
                    next_word()
                    continue

        elif in_quote == "'":
            if c == "'":
                in_quote = False

        elif in_quote == "\"":
            if escaped:
                if c not in escaped_in_double_quotes:
                    # as per the spec, these chars retain the backslash, so
                    # append that to cw before we append c as described below
                    cw = "\\" if cw is None else cw + "\\"
                escaped = False
            else:
                if c in special_in_double_quotes:
                    raise ValueError("not a shell-glob pattern: %s" % s)
                elif c == "\\":
                    escaped = True
                elif c == "\"":
                    in_quote = False

        else:
            assert False

        # append c onto cw. we do this uncondtionally because we're only
        # sanitising the string and not parsing it; we want to keep it in a
        # form that the shell can parse
        cw = c if cw is None else cw + c

    if in_quote or escaped:
        raise ValueError("unclosed escape or quote: %s" % s)

    next_word()

    words = ["./" + w for w in words] if forcerel else words
    return " ".join(words) if join else words


if __name__ == "__main__":
    import sys
    for a in sys.argv[1:]:
        print(sanitize_globs(a))
