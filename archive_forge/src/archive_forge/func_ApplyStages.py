from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import List, Optional, Dict
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.core.console import progress_tracker
def ApplyStages(resource_types: Optional[List[str]]=None) -> Dict[str, progress_tracker.Stage]:
    """Returns the progress tracker Stages for apply command.

  Args:
    resource_types: set of resource type strings to deploy.

  Returns:
    array of progress_tracker.Stage
  """
    stages = {UPDATE_APPLICATION: progress_tracker.Stage('Saving Configuration...', key=UPDATE_APPLICATION), CREATE_DEPLOYMENT: progress_tracker.Stage('Actuating Configuration...', key=CREATE_DEPLOYMENT)}
    if resource_types:
        deploy_stages = _DeployStages(resource_types, 'Configuring ')
        stages.update(deploy_stages)
    return stages