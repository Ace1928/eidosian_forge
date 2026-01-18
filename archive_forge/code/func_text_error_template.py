import sys
import traceback
from mako import compat
from mako import util
def text_error_template(lookup=None):
    """Provides a template that renders a stack trace in a similar format to
    the Python interpreter, substituting source template filenames, line
    numbers and code for that of the originating source template, as
    applicable.

    """
    import mako.template
    return mako.template.Template('\n<%page args="error=None, traceback=None"/>\n<%!\n    from mako.exceptions import RichTraceback\n%>\\\n<%\n    tback = RichTraceback(error=error, traceback=traceback)\n%>\\\nTraceback (most recent call last):\n% for (filename, lineno, function, line) in tback.traceback:\n  File "${filename}", line ${lineno}, in ${function or \'?\'}\n    ${line | trim}\n% endfor\n${tback.errorname}: ${tback.message}\n')