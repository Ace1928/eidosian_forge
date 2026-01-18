from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import threading
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
Initializes task.

    Args:
      source_resource (ObjectResource): Must contain the full path of object to
        download, including bucket. Directories will not be accepted. Does not
        need to contain metadata.
      destination_resource (resource_reference.Resource): Target resource to
        copy to. In this case, it contains the path of the destination stream or
        '-' for stdout.
      download_stream (stream): Reusable stream to write download to.
      print_created_message (bool): See parent class.
      print_source_version (bool): See parent class.
      show_url (bool): Says whether or not to print the header before each
        object's content
      start_byte (int): The byte index to start streaming from.
      end_byte (int|None): The byte index to stop streaming from.
      user_request_args (UserRequestArgs|None): See parent class.
      verbose (bool): See parent class.
    