from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
Initializes task.

    Args:
      bucket_url (storage_url.CloudUrl): URL of bucket that notification
        configuration exists on.
      notification_id (str): Name of the notification configuration (integer as
        string).
    