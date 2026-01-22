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
def ToCamelCase(name: str) -> str:
    """Turns a kebab case name into camel case.

  Args:
    name: the name string

  Returns:
    the string in camel case

  """
    pascal_case = name.replace('-', ' ').title().replace(' ', '')
    return pascal_case[0].lower() + pascal_case[1:]