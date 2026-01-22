from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.core.console import progress_tracker
def ServiceStages(include_iam_policy_set=False, include_route=True, include_build=False, include_create_repo=False, include_create_revision=True):
    """Return the progress tracker Stages for conditions of a Service."""
    stages = []
    if include_create_repo:
        stages.append(_CreateRepoStage())
    if include_build:
        stages.append(_UploadSourceStage())
        stages.append(_BuildContainerStage())
    if include_create_revision:
        stages.append(progress_tracker.Stage('Creating Revision...', key=SERVICE_CONFIGURATIONS_READY))
    if include_route:
        stages.append(_NewRoutingTrafficStage())
    if include_iam_policy_set:
        stages.append(progress_tracker.Stage('Setting IAM Policy...', key=SERVICE_IAM_POLICY_SET))
    return stages