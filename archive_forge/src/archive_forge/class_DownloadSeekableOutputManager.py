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
class DownloadSeekableOutputManager(DownloadOutputManager):

    @classmethod
    def is_compatible(cls, download_target, osutil):
        return seekable(download_target)

    def get_fileobj_for_io_writes(self, transfer_future):
        return transfer_future.meta.call_args.fileobj

    def get_final_io_task(self):
        return CompleteDownloadNOOPTask(transfer_coordinator=self._transfer_coordinator)