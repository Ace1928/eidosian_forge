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
def _UndeployResource(self, name, delete_selector, tracker, tracker_update_func=None):
    """Undeploy a resource.

    Args:
      name: name of the resource
      delete_selector: The selector for the undeploy operation.
      tracker: StagedProgressTracker, to report on the progress.
      tracker_update_func: optional custom fn to update the tracker.
    """
    tracker.StartStage(stages.UNDEPLOY_RESOURCE)
    self._CreateDeployment(appname=_DEFAULT_APP_NAME, tracker=tracker, tracker_update_func=tracker_update_func, delete_selector=delete_selector)
    tracker.CompleteStage(stages.UNDEPLOY_RESOURCE)
    app = self.GetDefaultApp()
    resource = self._FindResource(app.config, name)
    if resource:
        app.config.resourceList.remove(resource)
    tracker.StartStage(stages.CLEANUP_CONFIGURATION)
    self._UpdateApplication(appname=_DEFAULT_APP_NAME, appconfig=app.config, etag=app.etag)
    tracker.CompleteStage(stages.CLEANUP_CONFIGURATION)