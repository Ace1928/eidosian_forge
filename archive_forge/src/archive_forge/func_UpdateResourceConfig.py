from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Dict, Iterable, List, Optional, Set, TypedDict
import uuid
from apitools.base.py import encoding
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
def UpdateResourceConfig(self, parameters: Dict[str, str], resource: runapps_v1alpha1_messages.Resource) -> List[str]:
    """Updates config according to the parameters.

    Each TypeKit should override this method to update the resource config
    specific to the need of the typekit.

    Args:
      parameters: parameters from the command
      resource: the resource object of the integration

    Returns:
      list of service names referred in parameters.
    """
    metadata = self._type_metadata
    config_dict = {}
    if resource.config:
        config_dict = encoding.MessageToDict(resource.config)
    for param in metadata.parameters:
        param_value = parameters.get(param.name)
        if param_value:
            if param.data_type == 'int':
                config_dict[param.config_name] = int(param_value)
            elif param.data_type == 'number':
                config_dict[param.config_name] = float(param_value)
            else:
                config_dict[param.config_name] = param_value
    resource.config = encoding.DictToMessage(config_dict, runapps_v1alpha1_messages.Resource.ConfigValue)
    return []