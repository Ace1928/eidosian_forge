from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import pkgutil
import textwrap
from googlecloudsdk import api_lib
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_transform
import six
from six.moves import range  # pylint: disable=redefined-builtin
def _AppendLine(lines, line, paragraph):
    """Appends line to lines handling list markdown.

  Args:
    lines: The lines to append to.
    line: The line to append.
    paragraph: Start a new paragraph if True.

  Returns:
    The new paragraph value. This will always be False.
  """
    if paragraph:
        paragraph = False
        _AppendParagraph(lines)
    elif lines and (not lines[-1].endswith('\n')):
        lines.append(' ')
    if line.startswith('* ') or line.endswith('::'):
        if lines and (not lines[-1].endswith('\n')):
            lines.append('\n')
        lines.append(line)
        lines.append('\n')
    else:
        lines.append(line)
    return paragraph