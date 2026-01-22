from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
class DeleteObjectTask(CloudDeleteTask):
    """Task to delete an object."""

    def _make_delete_api_call(self, client, request_config):
        client.delete_object(self._url, request_config)