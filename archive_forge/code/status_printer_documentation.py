from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.kuberun import pretty_print
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import resource_printer
from six.moves.urllib_parse import urlparse
Transform ComponentStatus into the output structure of marker classes.

    Args:
      record: a dict object

    Returns:
      lines formatted for output
    