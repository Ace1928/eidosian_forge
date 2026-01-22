from the server back to the client.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_exceptions
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
def Quote(self, value, always=False):
    """Returns value or value "..." quoted with C-style escapes if needed.

    Args:
      value: The string value to quote if needed.
      always: Always quote non-numeric value if True.

    Returns:
      A string: value or value "..." quoted with C-style escapes if needed or
      requested.
    """
    try:
        return str(int(value))
    except ValueError:
        pass
    try:
        return str(float(value))
    except ValueError:
        pass
    chars = []
    enclose = always
    escaped = False
    for c in value:
        if escaped:
            escaped = False
        elif c == '\\':
            chars.append(c)
            chars.append(c)
            escaped = True
            enclose = True
        elif c == '"':
            chars.append('\\')
            enclose = True
        elif c.isspace() or c == "'":
            enclose = True
        chars.append(c)
    string = ''.join(chars)
    return '"{string}"'.format(string=string) if enclose else string