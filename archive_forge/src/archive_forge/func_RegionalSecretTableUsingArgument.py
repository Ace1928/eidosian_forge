from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import parser_arguments
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.secrets import args as secrets_args
def RegionalSecretTableUsingArgument(args: parser_extensions.Namespace, api_version: str='v1'):
    """Table format to display regional secrets.

  Args:
    args: arguments interceptor
    api_version: api version to be included in resource name
  """
    args.GetDisplayInfo().AddFormat(_REGIONAL_SECRET_TABLE)
    args.GetDisplayInfo().AddTransforms(_SECRET_TRANSFORMS)
    args.GetDisplayInfo().AddUriFunc(secrets_args.MakeGetUriFunc('secretmanager.projects.locations.secrets', api_version=api_version))