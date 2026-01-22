from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import encoding
import six
class GetPrinter(ValuePrinter):
    """A printer for printing value data with transforms disabled.

  Equivalent to the *value[no-transforms]* format. Default transforms are
  not applied to the displayed values.
  """

    def __init__(self, *args, **kwargs):
        super(GetPrinter, self).__init__(*args, ignore_default_transforms=True, **kwargs)