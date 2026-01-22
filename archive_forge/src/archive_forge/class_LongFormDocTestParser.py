import subprocess
import doctest
import os
import sys
import shutil
import re
import cgi
import rfc822
from io import StringIO
from paste.util import PySourceColor
class LongFormDocTestParser(doctest.DocTestParser):
    """
    This parser recognizes some reST comments as commands, without
    prompts or expected output, like:

    .. run:

        do_this(...
        ...)
    """
    _EXAMPLE_RE = re.compile("\n        # Source consists of a PS1 line followed by zero or more PS2 lines.\n        (?: (?P<source>\n                (?:^(?P<indent> [ ]*) >>>    .*)    # PS1 line\n                (?:\\n           [ ]*  \\.\\.\\. .*)*)  # PS2 lines\n            \\n?\n            # Want consists of any non-blank lines that do not start with PS1.\n            (?P<want> (?:(?![ ]*$)    # Not a blank line\n                         (?![ ]*>>>)  # Not a line starting with PS1\n                         .*$\\n?       # But any other line\n                      )*))\n        |\n        (?: # This is for longer commands that are prefixed with a reST\n            # comment like '.. run:' (two colons makes that a directive).\n            # These commands cannot have any output.\n\n            (?:^\\.\\.[ ]*(?P<run>run):[ ]*\\n) # Leading command/command\n            (?:[ ]*\\n)?         # Blank line following\n            (?P<runsource>\n                (?:(?P<runindent> [ ]+)[^ ].*$)\n                (?:\\n [ ]+ .*)*)\n            )\n        |\n        (?: # This is for shell commands\n\n            (?P<shellsource>\n                (?:^(P<shellindent> [ ]*) [$] .*)   # Shell line\n                (?:\\n               [ ]*  [>] .*)*) # Continuation\n            \\n?\n            # Want consists of any non-blank lines that do not start with $\n            (?P<shellwant> (?:(?![ ]*$)\n                              (?![ ]*[$]$)\n                              .*$\\n?\n                           )*))\n        ", re.MULTILINE | re.VERBOSE)

    def _parse_example(self, m, name, lineno):
        """
        Given a regular expression match from `_EXAMPLE_RE` (`m`),
        return a pair `(source, want)`, where `source` is the matched
        example's source code (with prompts and indentation stripped);
        and `want` is the example's expected output (with indentation
        stripped).

        `name` is the string's name, and `lineno` is the line number
        where the example starts; both are used for error messages.

        >>> def parseit(s):
        ...     p = LongFormDocTestParser()
        ...     return p._parse_example(p._EXAMPLE_RE.search(s), '<string>', 1)
        >>> parseit('>>> 1\\n1')
        ('1', {}, '1', None)
        >>> parseit('>>> (1\\n... +1)\\n2')
        ('(1\\n+1)', {}, '2', None)
        >>> parseit('.. run:\\n\\n    test1\\n    test2\\n')
        ('test1\\ntest2', {}, '', None)
        """
        runner = m.group('run') or ''
        indent = len(m.group('%sindent' % runner))
        source_lines = m.group('%ssource' % runner).split('\n')
        if runner:
            self._check_prefix(source_lines[1:], ' ' * indent, name, lineno)
        else:
            self._check_prompt_blank(source_lines, indent, name, lineno)
            self._check_prefix(source_lines[2:], ' ' * indent + '.', name, lineno)
        if runner:
            source = '\n'.join([sl[indent:] for sl in source_lines])
        else:
            source = '\n'.join([sl[indent + 4:] for sl in source_lines])
        if runner:
            want = ''
            exc_msg = None
        else:
            want = m.group('want')
            want_lines = want.split('\n')
            if len(want_lines) > 1 and re.match(' *$', want_lines[-1]):
                del want_lines[-1]
            self._check_prefix(want_lines, ' ' * indent, name, lineno + len(source_lines))
            want = '\n'.join([wl[indent:] for wl in want_lines])
            m = self._EXCEPTION_RE.match(want)
            if m:
                exc_msg = m.group('msg')
            else:
                exc_msg = None
        options = self._find_options(source, name, lineno)
        return (source, options, want, exc_msg)

    def parse(self, string, name='<string>'):
        """
        Divide the given string into examples and intervening text,
        and return them as a list of alternating Examples and strings.
        Line numbers for the Examples are 0-based.  The optional
        argument `name` is a name identifying this string, and is only
        used for error messages.
        """
        string = string.expandtabs()
        min_indent = self._min_indent(string)
        if min_indent > 0:
            string = '\n'.join([l[min_indent:] for l in string.split('\n')])
        output = []
        charno, lineno = (0, 0)
        for m in self._EXAMPLE_RE.finditer(string):
            output.append(string[charno:m.start()])
            lineno += string.count('\n', charno, m.start())
            source, options, want, exc_msg = self._parse_example(m, name, lineno)
            if not self._IS_BLANK_OR_COMMENT(source):
                output.append(doctest.Example(source, want, exc_msg, lineno=lineno, indent=min_indent + len(m.group('indent') or m.group('runindent')), options=options))
            lineno += string.count('\n', m.start(), m.end())
            charno = m.end()
        output.append(string[charno:])
        return output