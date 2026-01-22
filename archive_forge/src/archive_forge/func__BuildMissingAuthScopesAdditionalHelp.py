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
def _BuildMissingAuthScopesAdditionalHelp(known_exc):
    """Additional help when missing authentication scopes.

  When authenticated using user credentials and service account credentials
  locally, the requested scopes (googlecloudsdk.core.config.CLOUDSDK_SCOPES)
  should be enough to run gcloud commands. If users run gcloud from a GCE VM,
  the scopes of the default service account is customizable during vm creation.
  It is possible that the default service account does not have required scopes.

  Args:
    known_exc: googlecloudsdk.api_lib.util.exceptions.HttpException, The
     exception to handle.
  Returns:
    A HttpExceptionAdditionalHelp object.
  """
    error_message_signature = 'Request had insufficient authentication scopes'
    help_message = 'If you are in a compute engine VM, it is likely that the specified scopes during VM creation are not enough to run this command.\nSee https://cloud.google.com/compute/docs/access/service-accounts#accesscopesiam for more information about access scopes.\nSee https://cloud.google.com/compute/docs/access/create-enable-service-accounts-for-instances#changeserviceaccountandscopes for how to update access scopes of the VM.'
    return HttpExceptionAdditionalHelp(known_exc, error_message_signature, help_message)