from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import shlex
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.util import platforms
import six
Dispatches to the specific config printer.

    Nothing is printed if the record is not a JSON-serializable dict.

    Args:
      record: A JSON-serializable dict.
      delimit: Ignored.
    