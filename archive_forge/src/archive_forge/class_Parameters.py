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
class Parameters:
    """Each integration has a list of parameters that are stored in this class.

  Attributes:
    name: Name of the parameter.
    description: Explanation of the parameter that is visible to the
      customer.
    data_type: Denotes what values are acceptable for the parameter.
    update_allowed: If false, the param can not be provided in an update
      command.
    required:  If true, the param must be provided on a create command.
    hidden: If true, the param will not show up in error messages, but can
      be provided by the user.
    create_allowed: If false, the param cannot be provided on a create
      command.
    default: The value provided for the param if the user has not provided one.
    config_name: The name of the associated field in the config. If not
      provided, it will default to camelcase of `name`.
    label: The descriptive name of the param.
  """

    def __init__(self, name: str, description: str, data_type: str, update_allowed: bool=True, required: bool=False, hidden: bool=False, create_allowed: bool=True, default: Optional[object]=None, config_name: Optional[str]=None, label: Optional[str]=None):
        self.name = name
        self.config_name = config_name if config_name else ToCamelCase(name)
        self.description = description
        self.data_type = data_type
        self.update_allowed = update_allowed
        self.required = required
        self.hidden = hidden
        self.create_allowed = create_allowed
        self.default = default
        self.label = label