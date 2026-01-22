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
def GetFormatRegistry(hidden=False):
    """Returns the (format-name => Printer) format registry dictionary.

  Args:
    hidden: bool, if True, include the hidden formatters.

  Returns:
    The (format-name => Printer) format registry dictionary.
  """
    return {format_name: _FORMATTERS[format_name] for format_name in _FORMATTERS if hidden or format_name not in _HIDDEN_FORMATTERS}