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
class DeferQueue(object):
    """IO queue that defers write requests until they are queued sequentially.

    This class is used to track IO data for a *single* fileobj.

    You can send data to this queue, and it will defer any IO write requests
    until it has the next contiguous block available (starting at 0).

    """

    def __init__(self):
        self._writes = []
        self._pending_offsets = set()
        self._next_offset = 0

    def request_writes(self, offset, data):
        """Request any available writes given new incoming data.

        You call this method by providing new data along with the
        offset associated with the data.  If that new data unlocks
        any contiguous writes that can now be submitted, this
        method will return all applicable writes.

        This is done with 1 method call so you don't have to
        make two method calls (put(), get()) which acquires a lock
        each method call.

        """
        if offset < self._next_offset:
            return []
        writes = []
        if offset in self._pending_offsets:
            return []
        heapq.heappush(self._writes, (offset, data))
        self._pending_offsets.add(offset)
        while self._writes and self._writes[0][0] == self._next_offset:
            next_write = heapq.heappop(self._writes)
            writes.append({'offset': next_write[0], 'data': next_write[1]})
            self._pending_offsets.remove(next_write[0])
            self._next_offset += len(next_write[1])
        return writes