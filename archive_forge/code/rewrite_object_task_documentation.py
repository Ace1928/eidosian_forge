from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks.objects import patch_object_task
from googlecloudsdk.core import log
Initializes task.

    Args:
      object_resource (resource_reference.ObjectResource): The object to update.
      user_request_args (UserRequestArgs|None): Describes metadata updates to
        perform.
    