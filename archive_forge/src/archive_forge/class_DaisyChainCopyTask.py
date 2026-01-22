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
class DaisyChainCopyTask(copy_util.ObjectCopyTaskWithExitHandler):
    """Represents an operation to copy by downloading and uploading.

  This task downloads from one cloud location and uplaods to another cloud
  location by keeping an in-memory buffer.
  """

    def __init__(self, source_resource, destination_resource, delete_source=False, posix_to_set=None, print_created_message=False, print_source_version=False, user_request_args=None, verbose=False):
        """Initializes task.

    Args:
      source_resource (resource_reference.ObjectResource): Must contain the full
        object path of existing object. Directories will not be accepted.
      destination_resource (resource_reference.UnknownResource): Must contain
        the full object path. Object may not exist yet. Existing objects at the
        this location will be overwritten. Directories will not be accepted.
      delete_source (bool): If copy completes successfully, delete the source
        object afterwards.
      posix_to_set (PosixAttributes|None): See parent class.
      print_created_message (bool): See parent class.
      print_source_version (bool): See parent class.
      user_request_args (UserRequestArgs|None): See parent class.
      verbose (bool): See parent class.
    """
        super(DaisyChainCopyTask, self).__init__(source_resource, destination_resource, posix_to_set=posix_to_set, print_created_message=print_created_message, print_source_version=print_source_version, user_request_args=user_request_args, verbose=verbose)
        if not isinstance(source_resource.storage_url, storage_url.CloudUrl) or not isinstance(destination_resource.storage_url, storage_url.CloudUrl):
            raise errors.Error('DaisyChainCopyTask is for copies between cloud providers.')
        self._delete_source = delete_source
        self.parallel_processing_key = self._destination_resource.storage_url.url_string

    def _get_md5_hash(self):
        """Returns the MD5 Hash if present and hash validation is requested."""
        if properties.VALUES.storage.check_hashes.Get() == properties.CheckHashes.NEVER.value:
            return None
        if self._source_resource.md5_hash is None:
            log.warning('Found no hashes to validate object downloaded from %s and uploaded to %s. Integrity cannot be assured without hashes.', self._source_resource, self._destination_resource)
        return self._source_resource.md5_hash

    def _gapfill_request_config_field(self, resource_args, request_config_field_name, source_resource_field_name):
        request_config_value = getattr(resource_args, request_config_field_name, None)
        if request_config_value is None:
            setattr(resource_args, request_config_field_name, getattr(self._source_resource, source_resource_field_name))

    def _populate_request_config_with_resource_values(self, request_config):
        resource_args = request_config.resource_args
        self._gapfill_request_config_field(resource_args, 'cache_control', 'cache_control')
        self._gapfill_request_config_field(resource_args, 'content_disposition', 'content_disposition')
        self._gapfill_request_config_field(resource_args, 'content_encoding', 'content_encoding')
        self._gapfill_request_config_field(resource_args, 'content_language', 'content_language')
        self._gapfill_request_config_field(resource_args, 'content_type', 'content_type')
        self._gapfill_request_config_field(resource_args, 'custom_time', 'custom_time')
        self._gapfill_request_config_field(resource_args, 'md5_hash', 'md5_hash')

    def execute(self, task_status_queue=None):
        """Copies file by downloading and uploading in parallel."""
        destination_client = api_factory.get_api(self._destination_resource.storage_url.scheme)
        if copy_util.check_for_cloud_clobber(self._user_request_args, destination_client, self._destination_resource):
            log.status.Print(copy_util.get_no_clobber_message(self._destination_resource.storage_url))
            if self._send_manifest_messages:
                manifest_util.send_skip_message(task_status_queue, self._source_resource, self._destination_resource, copy_util.get_no_clobber_message(self._destination_resource.storage_url))
            return
        progress_callback = progress_callbacks.FilesAndBytesProgressCallback(status_queue=task_status_queue, offset=0, length=self._source_resource.size, source_url=self._source_resource.storage_url, destination_url=self._destination_resource.storage_url, operation_name=task_status.OperationName.DAISY_CHAIN_COPYING, process_id=os.getpid(), thread_id=threading.get_ident())
        buffer_controller = BufferController(self._source_resource, self._destination_resource.storage_url.scheme, self._user_request_args, progress_callback)
        buffer_controller.start_download_thread()
        content_type = self._source_resource.content_type or request_config_factory.DEFAULT_CONTENT_TYPE
        request_config = request_config_factory.get_request_config(self._destination_resource.storage_url, content_type=content_type, md5_hash=self._get_md5_hash(), size=self._source_resource.size, user_request_args=self._user_request_args)
        self._populate_request_config_with_resource_values(request_config)
        result_resource = None
        try:
            upload_strategy = upload_util.get_upload_strategy(api=destination_client, object_length=self._source_resource.size)
            result_resource = destination_client.upload_object(buffer_controller.readable_stream, self._destination_resource, request_config, posix_to_set=self._posix_to_set, source_resource=self._source_resource, upload_strategy=upload_strategy)
        except _AbruptShutdownError:
            pass
        except Exception as e:
            buffer_controller.shutdown(e)
        buffer_controller.wait_for_download_thread_to_terminate()
        buffer_controller.readable_stream.close()
        if buffer_controller.exception_raised:
            raise buffer_controller.exception_raised
        if result_resource:
            self._print_created_message_if_requested(result_resource)
            if self._send_manifest_messages:
                manifest_util.send_success_message(task_status_queue, self._source_resource, self._destination_resource, md5_hash=result_resource.md5_hash)
        if self._delete_source:
            return task.Output(additional_task_iterators=[[delete_task.DeleteObjectTask(self._source_resource.storage_url)]], messages=None)