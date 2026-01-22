from within calliope.
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
from functools import wraps
import os
import sys
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_attr_os
from googlecloudsdk.core.credentials import exceptions as creds_exceptions
import six
def _BuildMissingServiceUsePermissionAdditionalHelp(known_exc):
    """Additional help when missing the 'serviceusage.services.use' permission.

  Args:
    known_exc: googlecloudsdk.api_lib.util.exceptions.HttpException, The
     exception to handle.
  Returns:
    A HttpExceptionAdditionalHelp object.
  """
    error_message_signature = 'Grant the caller the Owner or Editor role, or a custom role with the serviceusage.services.use permission'
    help_message = 'If you want to invoke the command from a project different from the target resource project, use `--billing-project` or `{}` property.'.format(properties.VALUES.billing.quota_project)
    return HttpExceptionAdditionalHelp(known_exc, error_message_signature, help_message)