from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List, Optional, Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.core.console import progress_tracker
def _DeployStages(resource_types, stage_prefix):
    """Appends a deploy stage for each resource type in match_type_names.

  Args:
    resource_types: The set of resource type strings in the stage.
    stage_prefix: string. The prefix to add to the stage message.

  Returns:
    dict of stage key to progress_tracker Stage.
  """
    if not resource_types:
        return {}
    stages = {}
    for resource_type in resource_types:
        message = stage_prefix + _TypeToDescriptiveName(resource_type) + '...'
        stages[StageKeyForResourceDeployment(resource_type)] = progress_tracker.Stage(message, key=StageKeyForResourceDeployment(resource_type))
    return stages