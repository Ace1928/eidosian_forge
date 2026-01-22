from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
import frozendict
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.cloudresourcemanager import projects_util as projects_api_util
from googlecloudsdk.api_lib.functions.v2 import exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import encoding as encoder
from googlecloudsdk.core.util import retry
import six
def _GetOperationAndLogProgress(client, request, tracker, messages):
    """Returns a Boolean indicating whether the request has completed."""
    operation = client.projects_locations_operations.Get(request)
    if operation.error:
        raise exceptions.StatusToFunctionsError(operation.error, error_message=OperationErrorToString(operation.error))
    operation_metadata = _GetOperationMetadata(messages, operation)
    for stage in operation_metadata.stages:
        stage_in_progress = stage.state is GetStage(messages).StateValueValuesEnum.IN_PROGRESS
        stage_complete = stage.state is GetStage(messages).StateValueValuesEnum.COMPLETE
        if not stage_in_progress and (not stage_complete):
            continue
        stage_key = str(stage.name)
        if tracker.IsComplete(stage_key):
            continue
        if tracker.IsWaiting(stage_key):
            tracker.StartStage(stage_key)
        stage_message = stage.message or ''
        if stage_in_progress:
            stage_message = (stage_message or 'In progress') + '... '
        else:
            stage_message = ''
        if stage.resourceUri and stage_key == 'BUILD':
            stage_message += 'Logs are available at [{}]'.format(stage.resourceUri)
        tracker.UpdateStage(stage_key, stage_message)
        if stage_complete:
            if stage.stateMessages:
                tracker.CompleteStageWithWarnings(stage_key, GetStateMessagesStrings(stage.stateMessages))
            else:
                tracker.CompleteStage(stage_key)
    return operation