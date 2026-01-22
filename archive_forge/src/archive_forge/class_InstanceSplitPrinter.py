from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import traffic_pair
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
class InstanceSplitPrinter(cp.CustomPrinterBase):
    """Prints a worker's instance split in a custom human-readable format."""

    def Print(self, resources, single=False, intermediate=False):
        """Overrides ResourcePrinter.Print to set single=True."""
        super(InstanceSplitPrinter, self).Print(resources, True, intermediate)

    def Transform(self, record):
        """Transforms a List[TrafficTargetPair] into a marker class format."""
        return _TransformInstanceSplitPairs(record)