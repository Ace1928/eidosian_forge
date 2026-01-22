from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
import re
from typing import Any, Optional
from apitools.base.py import encoding
from googlecloudsdk.command_lib.run.integrations.formatters import base
from googlecloudsdk.command_lib.run.integrations.formatters import states
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import custom_printer_base as cp
from googlecloudsdk.core.resource import yaml_printer as yp
Call to action to use generated environment variables.

    If the resource state is not ACTIVE then the resource is not ready for
    use and the call to action will not be shown.

    It supports simple template value subsitution. Supported keys are:
    %%project%%: the name of the project
    %%region%%: the region
    %%config.X%%: the attribute from Resource's config with key 'X'
    %%status.X%%: the attribute from ResourceStatus' extraDetails with key 'X'

    Args:
      record: integration_printer.Record class that just holds data.

    Returns:
      A formatted string of the call to action message,
      or None if no call to action is required.
    