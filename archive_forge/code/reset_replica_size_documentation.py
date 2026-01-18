from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import exceptions
from googlecloudsdk.api_lib.sql import operations
from googlecloudsdk.api_lib.sql import validate
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.sql import flags
from googlecloudsdk.core import properties
import six.moves.http_client
Resets a Cloud SQL read replica storage size to its primary storage size.

    Args:
      args: argparse.Namespace, The arguments with which this command was
        invoked.

    Returns:
      A dict object representing the operations resource describing the reset
      replica size operation if the reset was successful.

    Raises:
      HttpException: A http error response was received while executing an api
          request.
      ResourceNotFoundError: The SQL instance isn't found.
    