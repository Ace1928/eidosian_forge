from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import encoding
import six
def _QuoteField(self, field):
    """Returns field quoted by self._quote if necessary.

    The Python 2.7 csv module does not support unicode "yet". What are they
    waiting for?

    Args:
      field: The unicode string to quote.

    Returns:
      field quoted by self._quote if necessary.
    """
    if not field or not self._quote:
        return field
    if not (self._delimiter in field or self._quote in field or self._separator in field or (self._terminator in field) or field[0].isspace() or field[-1].isspace()):
        return field
    return self._quote + field.replace(self._quote, self._quote * 2) + self._quote