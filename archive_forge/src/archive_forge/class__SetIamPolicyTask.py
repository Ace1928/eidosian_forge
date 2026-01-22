from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage.tasks import task
class _SetIamPolicyTask(task.Task):
    """Base class for tasks that set IAM policies."""

    def __init__(self, url, policy):
        """Initializes task.

    Args:
      url (StorageUrl): Used to identify cloud resource to set policy on.
      policy (object): Provider-specific data type. Currently, only available
        for GCS so Apitools messages.Policy object. If supported for more
        providers in the future, use a generic container.
    """
        super(_SetIamPolicyTask, self).__init__()
        self._url = url
        self._policy = policy

    @abc.abstractmethod
    def _make_set_api_call(self, client):
        """Makes an API call to set the IAM policy. Overridden by children."""
        pass

    def execute(self, task_status_queue=None):
        """Executes task."""
        client = api_factory.get_api(self._url.scheme)
        new_policy = self._make_set_api_call(client)
        if task_status_queue:
            progress_callbacks.increment_count_callback(task_status_queue)
        return task.Output(additional_task_iterators=None, messages=[task.Message(task.Topic.SET_IAM_POLICY, payload=new_policy)])

    def __eq__(self, other):
        if not isinstance(other, _SetIamPolicyTask):
            return NotImplemented
        return self._url == other._url and self._policy == other._policy