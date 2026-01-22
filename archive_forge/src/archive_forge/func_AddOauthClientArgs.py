from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.iap import util as iap_api
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.iap import exceptions as iap_exc
from googlecloudsdk.core import properties
def AddOauthClientArgs(parser):
    """Adds OAuth client args.

  Args:
    parser: An argparse.ArgumentParser-like object. It is mocked out in order to
      capture some information, but behaves like an ArgumentParser.
  """
    group = parser.add_group()
    group.add_argument('--oauth2-client-id', required=True, help='OAuth 2.0 client ID to use.')
    group.add_argument('--oauth2-client-secret', required=True, help='OAuth 2.0 client secret to use.')