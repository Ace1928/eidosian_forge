from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import shlex
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.util import platforms
import six
def _PrintCategory(self, out, label, items):
    """Prints config items in the label category.

    Args:
      out: The output stream for this category.
      label: A list of label strings.
      items: The items to list for label, either dict iteritems, an enumerated
        list, or a scalar value.
    """
    top = io.StringIO()
    sub = io.StringIO()
    for name, value in sorted(items):
        name = six.text_type(name)
        try:
            values = six.iteritems(value)
            self._PrintCategory(sub, label + [name], values)
            continue
        except AttributeError:
            pass
        if value is None:
            top.write('{name} (unset)\n'.format(name=name))
        elif isinstance(value, list):
            self._PrintCategory(sub, label + [name], enumerate(value))
        else:
            top.write('{name} = {value}\n'.format(name=name, value=value))
    top_content = top.getvalue()
    sub_content = sub.getvalue()
    if label and (top_content or (sub_content and (not sub_content.startswith('[')))):
        out.write('[{0}]\n'.format('.'.join(label)))
    if top_content:
        out.write(top_content)
    if sub_content:
        out.write(sub_content)