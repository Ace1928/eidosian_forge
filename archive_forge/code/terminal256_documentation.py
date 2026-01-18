import sys
from pygments.formatter import Formatter
from pygments.console import codes
from pygments.style import ansicolors

    Format tokens with ANSI color sequences, for output in a true-color
    terminal or console.  Like in `TerminalFormatter` color sequences
    are terminated at newlines, so that paging the output works correctly.

    .. versionadded:: 2.1

    Options accepted:

    `style`
        The style to use, can be a string or a Style subclass (default:
        ``'default'``).
    