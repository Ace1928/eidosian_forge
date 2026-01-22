from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import json
import os
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
Updates the ECP config based on the passed in CLI arguments.

  Args:
    config_type: An ConfigType Enum that describes the type of ECP config.
    **kwargs: config parameters that will be updated. See
      go/enterprise-cert-config for valid variables.

  Only explicit args will overwrite existing values
  