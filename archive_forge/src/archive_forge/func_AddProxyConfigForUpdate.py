from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.gkemulticloud import util as api_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
def AddProxyConfigForUpdate(parser, noun):
    """Adds proxy configuration flags for update.

  Args:
    parser: The argparse.parser to add the arguments to.
    noun: The resource type to which the flags are applicable.
  """
    group = parser.add_group('Proxy config', mutex=True)
    update_proxy_group = group.add_group('Update existing proxy config parameters')
    AddProxySecretArn(update_proxy_group)
    AddProxySecretVersionId(update_proxy_group)
    AddClearProxyConfig(group, noun)