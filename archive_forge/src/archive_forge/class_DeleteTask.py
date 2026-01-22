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
class DeleteTask(task.Task):
    """Base class for tasks that delete a resource."""

    def __init__(self, url, user_request_args=None, verbose=True):
        """Initializes task.

    Args:
      url (storage_url.StorageUrl): URL of the resource to delete.
      user_request_args (UserRequestArgs|None): Values for RequestConfig.
      verbose (bool): If true, prints status messages. Otherwise, does not print
        anything.
    """
        super().__init__()
        self._url = url
        self._user_request_args = user_request_args
        self._verbose = verbose
        self.parallel_processing_key = url.url_string

    @abc.abstractmethod
    def _perform_deletion(self):
        """Deletes a resource. Overridden by children."""
        raise NotImplementedError

    def execute(self, task_status_queue=None):
        if self._verbose:
            log.status.Print('Removing {}...'.format(self._url))
        self._perform_deletion()
        if task_status_queue:
            progress_callbacks.increment_count_callback(task_status_queue)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._url == other._url and self._user_request_args == other._user_request_args and (self._verbose == other._verbose)