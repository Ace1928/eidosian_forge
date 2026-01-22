from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
from typing import List, Optional
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_client
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def GetTypeMetadataByResource(resource: runapps_v1alpha1_messages.Resource) -> Optional[TypeMetadata]:
    """Returns metadata associated to an integration type.

  Args:
    resource: the resource object

  Returns:
    If the integration does not exist or is not visible to the user,
    then None is returned.
  """
    return GetTypeMetadataByResourceType(resource.id.type)