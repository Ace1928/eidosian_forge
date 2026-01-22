from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.command_lib.tasks import app
from googlecloudsdk.command_lib.tasks import constants
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import http_encoding
import six  # pylint: disable=unused-import
from six.moves import filter  # pylint:disable=redefined-builtin
from six.moves import map  # pylint:disable=redefined-builtin
def _IsSpecifiedWrapper(arg):
    """Wrapper function for Namespace.IsSpecified function.

    We need this function to be support being able to modify certain queue
    attributes internally using `gcloud app deploy queue.yaml` without exposing
    the same functionality via `gcloud tasks queues create/update`.

    Args:
      arg: The argument we are trying to check if specified.

    Returns:
      True if the argument was specified at CLI invocation, False otherwise.
    """
    http_queue_args = ['http_uri_override', 'http_method_override', 'http_header_override', 'http_oauth_service_account_email_override', 'http_oauth_token_scope_override', 'http_oidc_service_account_email_override', 'http_oidc_token_audience_override']
    try:
        return specified_args_object.IsSpecified(arg)
    except parser_errors.UnknownDestinationException:
        if arg in ('max_burst_size', 'clear_max_burst_size') or any((flag in arg for flag in http_queue_args)):
            return False
        raise