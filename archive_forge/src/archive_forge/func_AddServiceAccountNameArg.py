from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import binascii
import re
import textwrap
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as gcloud_exceptions
from googlecloudsdk.command_lib.iam import completers
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
import six
def AddServiceAccountNameArg(parser, action='to act on'):
    """Adds the IAM service account name argument that supports tab completion.

  Args:
    parser: An argparse.ArgumentParser-like object to which we add the args.
    action: Action to display in the help message. Should be something like 'to
      act on' or a relative phrase like 'whose policy to get'.

  Raises:
    ArgumentError if one of the arguments is already defined in the parser.
  """
    parser.add_argument('service_account', metavar='SERVICE_ACCOUNT', type=GetIamAccountFormatValidator(), completer=completers.IamServiceAccountCompleter, help='The service account {}. The account should be formatted either as a numeric service account ID or as an email, like this: 123456789892843212345 or my-iam-account@somedomain.com.'.format(action))