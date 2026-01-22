from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import shlex
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.util import platforms
import six
@staticmethod
def _Prefix(prefix, name):
    """Returns a new prefix based on prefix and name."""
    if isinstance(name, int):
        name = 'I' + six.text_type(name)
    return prefix + name + '_'