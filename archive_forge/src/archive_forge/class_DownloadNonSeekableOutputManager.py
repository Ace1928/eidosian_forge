import logging
import threading
import heapq
from botocore.compat import six
from s3transfer.compat import seekable
from s3transfer.exceptions import RetriesExceededError
from s3transfer.futures import IN_MEMORY_DOWNLOAD_TAG
from s3transfer.utils import S3_RETRYABLE_DOWNLOAD_ERRORS
from s3transfer.utils import get_callbacks
from s3transfer.utils import invoke_progress_callbacks
from s3transfer.utils import calculate_num_parts
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import FunctionContainer
from s3transfer.utils import CountCallbackInvoker
from s3transfer.utils import StreamReaderProgress
from s3transfer.utils import DeferredOpenFile
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
class DownloadNonSeekableOutputManager(DownloadOutputManager):

    def __init__(self, osutil, transfer_coordinator, io_executor, defer_queue=None):
        super(DownloadNonSeekableOutputManager, self).__init__(osutil, transfer_coordinator, io_executor)
        if defer_queue is None:
            defer_queue = DeferQueue()
        self._defer_queue = defer_queue
        self._io_submit_lock = threading.Lock()

    @classmethod
    def is_compatible(cls, download_target, osutil):
        return hasattr(download_target, 'write')

    def get_download_task_tag(self):
        return IN_MEMORY_DOWNLOAD_TAG

    def get_fileobj_for_io_writes(self, transfer_future):
        return transfer_future.meta.call_args.fileobj

    def get_final_io_task(self):
        return CompleteDownloadNOOPTask(transfer_coordinator=self._transfer_coordinator)

    def queue_file_io_task(self, fileobj, data, offset):
        with self._io_submit_lock:
            writes = self._defer_queue.request_writes(offset, data)
            for write in writes:
                data = write['data']
                logger.debug('Queueing IO offset %s for fileobj: %s', write['offset'], fileobj)
                super(DownloadNonSeekableOutputManager, self).queue_file_io_task(fileobj, data, offset)

    def get_io_write_task(self, fileobj, data, offset):
        return IOStreamingWriteTask(self._transfer_coordinator, main_kwargs={'fileobj': fileobj, 'data': data})