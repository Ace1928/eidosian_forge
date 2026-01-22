from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.command_lib.storage import progress_callbacks
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage.tasks import task
from googlecloudsdk.core import log
class ResumeAnywhereCacheTask(task.Task):
    """Task for resuming an Anywhere Cache instance."""

    def __init__(self, bucket_name, anywhere_cache_id):
        """Initializes task."""
        super(ResumeAnywhereCacheTask, self).__init__()
        self._bucket_name = bucket_name
        self._anywhere_cache_id = anywhere_cache_id
        self.parallel_processing_key = '{}/{}'.format(bucket_name, anywhere_cache_id)

    def execute(self, task_status_queue=None):
        log.status.Print('Requesting to resume a cache instance of bucket gs://{} having anywhere_cache_id: {}'.format(self._bucket_name, self._anywhere_cache_id))
        provider = storage_url.ProviderPrefix.GCS
        api_factory.get_api(provider).resume_anywhere_cache(self._bucket_name, self._anywhere_cache_id)
        if task_status_queue:
            progress_callbacks.increment_count_callback(task_status_queue)

    def __eq__(self, other):
        if not isinstance(other, ResumeAnywhereCacheTask):
            return NotImplemented
        return self._bucket_name == other._bucket_name and self._anywhere_cache_id == other._anywhere_cache_id