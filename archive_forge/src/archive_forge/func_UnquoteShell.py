from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
def UnquoteShell(s):
    """Processes a quoted shell argument from the lexer.

  Args:
    s: the raw quoted string (ex: "\\"foo\\" \\\\ 'bar'")

  Returns:
    An argument as would be passed from a shell to a process it forks
    (ex: "foo" \\ 'bar').

  """
    return s