from the :func:`setup()` function.
import logging
import types
import docutils.nodes
import docutils.utils
from humanfriendly.deprecation import get_aliases
from humanfriendly.text import compact, dedent, format
from humanfriendly.usage import USAGE_MARKER, render_usage
def update_lines(lines, text):
    """Private helper for ``autodoc-process-docstring`` callbacks."""
    while lines:
        lines.pop()
    lines.extend(text.splitlines())