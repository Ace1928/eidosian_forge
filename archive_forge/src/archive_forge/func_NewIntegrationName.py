from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def NewIntegrationName(self, appconfig: runapps_v1alpha1_messages.Config) -> str:
    """Returns a name for a new integration.

    Args:
      appconfig: the application config

    Returns:
      str, a new name for the integration.
    """
    name = self._GenerateIntegrationNameCandidate(self.integration_type)
    existing_names = {res.id.name for res in appconfig.resourceList if res.id.type == self.resource_type}
    while name in existing_names:
        name = self._GenerateIntegrationNameCandidate(self.integration_type)
    return name