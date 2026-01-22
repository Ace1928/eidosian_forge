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
def _UpdateApplication(self, appname: str, appconfig: runapps_v1alpha1_messages.Config, etag: str):
    """Update Application config, waits for operation to finish.

    Args:
      appname:  name of the application.
      appconfig: config of the application.
      etag: the etag of the application if it's an incremental patch.
    """
    app_ref = self.GetAppRef(appname)
    application = self.messages.Application(name=appname, config=appconfig, etag=etag)
    is_patch = etag or api_utils.GetApplication(self._client, app_ref)
    if is_patch:
        operation = api_utils.PatchApplication(self._client, app_ref, application)
    else:
        operation = api_utils.CreateApplication(self._client, app_ref, application)
    api_utils.WaitForApplicationOperation(self._client, operation)