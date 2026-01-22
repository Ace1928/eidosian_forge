from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.domains import operations
from googlecloudsdk.api_lib.domains import registrations
from googlecloudsdk.command_lib.domains import flags
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def ParseRegisterNotices(notices):
    """Parses registration notices.

  Args:
    notices: list of notices (lowercase-strings).

  Returns:
    Pair (public privacy ack: bool, hsts ack: bool).
  """
    if not notices:
        return (False, False)
    return ('public-contact-data-acknowledgement' in notices, 'hsts-preloaded' in notices)