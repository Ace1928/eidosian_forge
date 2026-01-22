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
class CompleteDownloadNOOPTask(Task):
    """A NOOP task to serve as an indicator that the download is complete

    Note that the default for is_final is set to True because this should
    always be the last task.
    """

    def __init__(self, transfer_coordinator, main_kwargs=None, pending_main_kwargs=None, done_callbacks=None, is_final=True):
        super(CompleteDownloadNOOPTask, self).__init__(transfer_coordinator=transfer_coordinator, main_kwargs=main_kwargs, pending_main_kwargs=pending_main_kwargs, done_callbacks=done_callbacks, is_final=is_final)

    def _main(self):
        pass