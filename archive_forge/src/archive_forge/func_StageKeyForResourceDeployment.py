from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List, Optional, Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.core.console import progress_tracker
def StageKeyForResourceDeployment(resource_type):
    """Returns the stage key for the step that deploys a resource type.

  Args:
    resource_type: The resource type string.

  Returns:
    stage key for deployment of type.
  """
    return _DEPLOY_STAGE_PREFIX + resource_type