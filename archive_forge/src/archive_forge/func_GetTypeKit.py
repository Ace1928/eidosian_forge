from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.run.integrations.typekits import custom_domains_typekit
from googlecloudsdk.command_lib.run.integrations.typekits import redis_typekit
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def GetTypeKit(integration_type: str) -> base.TypeKit:
    """Returns a typekit for the given integration type.

  Args:
    integration_type: type of integration.

  Raises:
    ArgumentError: If the integration type is not supported.

  Returns:
    A typekit instance.
  """
    if integration_type == 'custom-domains':
        return custom_domains_typekit.CustomDomainsTypeKit(types_utils.GetTypeMetadata('custom-domains'))
    if integration_type == 'redis':
        return redis_typekit.RedisTypeKit(types_utils.GetTypeMetadata('redis'))
    typekit = types_utils.GetTypeMetadata(integration_type)
    if typekit:
        return base.TypeKit(typekit)
    raise exceptions.ArgumentError('Integration of type {} is not supported'.format(integration_type))