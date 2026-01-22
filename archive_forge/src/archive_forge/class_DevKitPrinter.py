from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.resource import custom_printer_base as cp
class DevKitPrinter(cp.CustomPrinterBase):
    """Prints the kuberun DevKit custom human-readable format.
  """

    def _ComponentTable(self, record):
        rows = [(x.name, str(x.event_input), x.description) for x in record.components]
        return cp.Table([('NAME', 'TAKES CE-INPUT', 'DESCRIPTION')] + rows)

    def Transform(self, record):
        """Transform a service into the output structure of marker classes."""
        return cp.Labeled([('Name', record.name), ('Version', record.version), ('Description', record.description), ('Supported Components', self._ComponentTable(record))])