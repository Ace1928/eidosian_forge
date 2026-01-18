from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
Initializes task.

    Args:
      object_resource (resource_reference.ObjectResource): Object to restore.
      user_request_args (UserRequestArgs|None): Contains restore settings.
    