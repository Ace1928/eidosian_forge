from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
class PatchAnywhereCacheTask(task.Task):
    """Updates an Anywhere Cache instance."""

    def __init__(self, bucket_name, anywhere_cache_id, admission_policy=None, ttl=None):
        """Initializes task.

    Args:
      bucket_name (str): The name of the bucket where the Anywhere Cache should
        be updated.
      anywhere_cache_id (str): Name of the zonal location where the Anywhere
        Cache should be updated.
      admission_policy (str|None): The cache admission policy decides for each
        cache miss, that is whether to insert the missed block or not.
      ttl (str|None): Cache entry time-to-live in seconds
    """
        super(PatchAnywhereCacheTask, self).__init__()
        self._bucket_name = bucket_name
        self._anywhere_cache_id = anywhere_cache_id
        self._admission_policy = admission_policy
        self._ttl = ttl
        self.parallel_processing_key = '{}/{}'.format(bucket_name, anywhere_cache_id)

    def execute(self, task_status_queue=None):
        log.status.Print('Updating a cache instance of bucket gs://{} having anywhere_cache_id {}'.format(self._bucket_name, self._anywhere_cache_id))
        provider = storage_url.ProviderPrefix.GCS
        response = api_factory.get_api(provider).patch_anywhere_cache(self._bucket_name, self._anywhere_cache_id, admission_policy=self._admission_policy, ttl=self._ttl)
        log.status.Print('Initiated the operation id: {} for updating a cache instance of bucket gs://{} having anywhere_cache_id {}'.format(response.name, self._bucket_name, self._anywhere_cache_id))
        if task_status_queue:
            progress_callbacks.increment_count_callback(task_status_queue)

    def __eq__(self, other):
        if not isinstance(other, PatchAnywhereCacheTask):
            return NotImplemented
        return self._bucket_name == other._bucket_name and self._anywhere_cache_id == other._anywhere_cache_id and (self._admission_policy == other._admission_policy) and (self._ttl == other._ttl)