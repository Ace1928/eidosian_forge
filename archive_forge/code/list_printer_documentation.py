from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.util import encoding
import six
Immediately prints the given record as a list item.

    Args:
      record: A JSON-serializable object.
      delimit: Prints resource delimiters if True.
    