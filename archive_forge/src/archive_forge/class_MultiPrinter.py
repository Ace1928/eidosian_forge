from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties as core_properties
from googlecloudsdk.core.resource import config_printer
from googlecloudsdk.core.resource import csv_printer
from googlecloudsdk.core.resource import diff_printer
from googlecloudsdk.core.resource import flattened_printer
from googlecloudsdk.core.resource import json_printer
from googlecloudsdk.core.resource import list_printer
from googlecloudsdk.core.resource import object_printer
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_printer_types as formats
from googlecloudsdk.core.resource import resource_projection_spec
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.resource import table_printer
from googlecloudsdk.core.resource import yaml_printer
class MultiPrinter(resource_printer_base.ResourcePrinter):
    """A printer that prints different formats for each projection key.

  Each projection key must have a subformat defined by the
  :format=FORMAT-STRING attribute. For example,

    `--format="multi(data:format=json, info:format='table[box](a, b, c)')"`

  formats the *data* field as JSON and the *info* field as a boxed table.

  Printer attributes:
    separator: Separator string to print between each format. If multiple
      resources are provided, the separator is also printed between each
      resource.
  """

    def __init__(self, *args, **kwargs):
        super(MultiPrinter, self).__init__(*args, **kwargs)
        self.columns = []
        for col in self.column_attributes.Columns():
            if not col.attribute.subformat:
                raise ProjectionFormatRequiredError('{key} requires format attribute.'.format(key=resource_lex.GetKeyName(col.key)))
            self.columns.append((col, Printer(col.attribute.subformat, out=self._out)))

    def _AddRecord(self, record, delimit=True):
        separator = self.attributes.get('separator', '')
        for i, (col, printer) in enumerate(self.columns):
            if i != 0 or delimit:
                self._out.write(separator)
            printer.Print(resource_property.Get(record, col.key))
        terminator = self.attributes.get('terminator', '')
        if terminator:
            self._out.write(terminator)