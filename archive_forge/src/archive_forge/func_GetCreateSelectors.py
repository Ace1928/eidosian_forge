from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.command_lib.run.integrations.typekits import base
def GetCreateSelectors(self, integration_name):
    """Returns create selectors for given integration and service.

    Args:
      integration_name: str, name of integration.

    Returns:
      list of dict typed names.
    """
    selectors = super(RedisTypeKit, self).GetCreateSelectors(integration_name)
    selectors.append({'type': 'vpc', 'name': '*'})
    return selectors