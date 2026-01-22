from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import files
def WriteToYamlFile(yaml_file, change):
    """Writes the given change in yaml format to the given file.

  Args:
    yaml_file: file, File into which the change should be written.
    change: Change, Change to be written out.
  """
    resource_printer.Print([change], print_format='yaml', out=yaml_file)