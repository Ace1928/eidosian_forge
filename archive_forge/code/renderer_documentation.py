from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import io
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_printer
import six
from six.moves import range  # pylint: disable=redefined-builtin
Renders a table.

    Nested tables are not supported.

    Args:
      table: A TableAttributes object.
      rows: A list of rows where each row is a list of column strings.
    