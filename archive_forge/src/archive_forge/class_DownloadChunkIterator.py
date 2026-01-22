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
class DownloadChunkIterator(object):

    def __init__(self, body, chunksize):
        """Iterator to chunk out a downloaded S3 stream

        :param body: A readable file-like object
        :param chunksize: The amount to read each time
        """
        self._body = body
        self._chunksize = chunksize
        self._num_reads = 0

    def __iter__(self):
        return self

    def __next__(self):
        chunk = self._body.read(self._chunksize)
        self._num_reads += 1
        if chunk:
            return chunk
        elif self._num_reads == 1:
            return chunk
        raise StopIteration()
    next = __next__