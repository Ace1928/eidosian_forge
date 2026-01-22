from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import functools
import os
import re
import sys
import textwrap
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.configurations import properties_file as prop_files_lib
from googlecloudsdk.core.docker import constants as const_lib
from googlecloudsdk.core.resource import resource_printer_types as formats
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.generated_clients.apis import apis_map
import six
def GetValueFromFeatureFlag(prop):
    """Gets the property value from the Feature Flags yaml.

  Args:
    prop: The property to get

  Returns:
    str, the value of the property, or None if it is not set.
  """
    from googlecloudsdk.core.feature_flags import config as feature_flags_config
    ff_config = feature_flags_config.GetFeatureFlagsConfig(VALUES.core.account.Get(), VALUES.core.project.Get())
    if ff_config:
        return Stringize(ff_config.Get(prop))
    return None