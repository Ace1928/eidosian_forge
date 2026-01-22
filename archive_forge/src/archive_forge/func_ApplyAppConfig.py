from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import datetime
import json
from typing import List, MutableSequence, Optional
from apitools.base.py import exceptions as api_exceptions
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.run.integrations import api_utils
from googlecloudsdk.api_lib.run.integrations import types_utils
from googlecloudsdk.api_lib.run.integrations import validator
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.run import connection_context
from googlecloudsdk.command_lib.run import flags as run_flags
from googlecloudsdk.command_lib.run import serverless_operations
from googlecloudsdk.command_lib.run.integrations import flags
from googlecloudsdk.command_lib.run.integrations import integration_list_printer
from googlecloudsdk.command_lib.run.integrations import messages_util
from googlecloudsdk.command_lib.run.integrations import stages
from googlecloudsdk.command_lib.run.integrations import typekits_util
from googlecloudsdk.command_lib.run.integrations.typekits import base
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.generated_clients.apis.runapps.v1alpha1 import runapps_v1alpha1_messages
import six
def ApplyAppConfig(self, tracker, appname: str, appconfig: runapps_v1alpha1_messages.Config, integration_name=None, deploy_message=None, match_type_names=None, intermediate_step=False, etag=None, tracker_update_func=None):
    """Applies the application config.

    Args:
      tracker: StagedProgressTracker, to report on the progress.
      appname:  name of the application.
      appconfig: config of the application.
      integration_name: name of the integration that's being updated.
      deploy_message: message to display when deployment in progress.
      match_type_names: array of type/name pairs used for create selector.
      intermediate_step: bool of whether this is an intermediate step.
      etag: the etag of the application if it's an incremental patch.
      tracker_update_func: optional custom fn to update the tracker.
    """
    tracker.StartStage(stages.UPDATE_APPLICATION)
    if integration_name:
        tracker.UpdateStage(stages.UPDATE_APPLICATION, messages_util.CheckStatusMessage(self._release_track, integration_name))
    try:
        self._UpdateApplication(appname, appconfig, etag)
    except api_exceptions.HttpConflictError as err:
        _HandleQueueingException(err)
    except exceptions.IntegrationsOperationError as err:
        tracker.FailStage(stages.UPDATE_APPLICATION, err)
    else:
        tracker.CompleteStage(stages.UPDATE_APPLICATION)
    if match_type_names is None:
        match_type_names = [{'type': '*', 'name': '*'}]
    create_selector = {'matchTypeNames': match_type_names}
    if not intermediate_step:
        tracker.UpdateHeaderMessage('Deployment started. This process will continue even if your terminal session is interrupted.')
    tracker.StartStage(stages.CREATE_DEPLOYMENT)
    if deploy_message:
        tracker.UpdateStage(stages.CREATE_DEPLOYMENT, deploy_message)
    try:
        self._CreateDeployment(appname, tracker, tracker_update_func=tracker_update_func, create_selector=create_selector)
    except api_exceptions.HttpConflictError as err:
        _HandleQueueingException(err)
    except exceptions.IntegrationsOperationError as err:
        tracker.FailStage(stages.CREATE_DEPLOYMENT, err)
    else:
        tracker.UpdateStage(stages.CREATE_DEPLOYMENT, '')
        tracker.CompleteStage(stages.CREATE_DEPLOYMENT)
    tracker.UpdateHeaderMessage('Done.')