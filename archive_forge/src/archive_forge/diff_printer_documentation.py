from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.core.resource import resource_printer_base
Immediately prints the first two columns of record as a unified diff.

    Records with less than 2 colums are silently ignored.

    Args:
      record: A JSON-serializable object.
      delimit: Prints resource delimiters if True.
    