from the :func:`setup()` function.
import logging
import types
import docutils.nodes
import docutils.utils
from humanfriendly.deprecation import get_aliases
from humanfriendly.text import compact, dedent, format
from humanfriendly.usage import USAGE_MARKER, render_usage
def usage_message_callback(app, what, name, obj, options, lines):
    """
    Reformat human friendly usage messages to reStructuredText_.

    Refer to :func:`enable_usage_formatting()` to enable the use of this
    function (you probably don't want to call :func:`usage_message_callback()`
    directly).

    This function implements a callback for ``autodoc-process-docstring`` that
    reformats module docstrings using :func:`.render_usage()` so that Sphinx
    doesn't mangle usage messages that were written to be human readable
    instead of machine readable. Only module docstrings whose first line starts
    with :data:`.USAGE_MARKER` are reformatted.

    The parameters expected by this function are those defined for Sphinx event
    callback functions (i.e. I'm not going to document them here :-).
    """
    if isinstance(obj, types.ModuleType) and lines:
        if lines[0].startswith(USAGE_MARKER):
            text = render_usage('\n'.join(lines))
            update_lines(lines, text)