from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import threading
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import request_config_factory
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import manifest_util
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.command_lib.storage.tasks import task_status
from googlecloudsdk.command_lib.storage.tasks.cp import copy_util
from googlecloudsdk.command_lib.storage.tasks.cp import upload_util
from googlecloudsdk.command_lib.storage.tasks.rm import delete_task
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _populate_request_config_with_resource_values(self, request_config):
    resource_args = request_config.resource_args
    self._gapfill_request_config_field(resource_args, 'cache_control', 'cache_control')
    self._gapfill_request_config_field(resource_args, 'content_disposition', 'content_disposition')
    self._gapfill_request_config_field(resource_args, 'content_encoding', 'content_encoding')
    self._gapfill_request_config_field(resource_args, 'content_language', 'content_language')
    self._gapfill_request_config_field(resource_args, 'content_type', 'content_type')
    self._gapfill_request_config_field(resource_args, 'custom_time', 'custom_time')
    self._gapfill_request_config_field(resource_args, 'md5_hash', 'md5_hash')