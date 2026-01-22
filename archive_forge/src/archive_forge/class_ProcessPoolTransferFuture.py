import collections
import contextlib
import logging
import multiprocessing
import threading
import signal
from copy import deepcopy
import botocore.session
from botocore.config import Config
from s3transfer.constants import MB
from s3transfer.constants import ALLOWED_DOWNLOAD_ARGS
from s3transfer.constants import PROCESS_USER_AGENT
from s3transfer.compat import MAXINT
from s3transfer.compat import BaseManager
from s3transfer.exceptions import CancelledError
from s3transfer.exceptions import RetriesExceededError
from s3transfer.futures import BaseTransferFuture
from s3transfer.futures import BaseTransferMeta
from s3transfer.utils import S3_RETRYABLE_DOWNLOAD_ERRORS
from s3transfer.utils import calculate_num_parts
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import OSUtils
from s3transfer.utils import CallArgs
class ProcessPoolTransferFuture(BaseTransferFuture):

    def __init__(self, monitor, meta):
        """The future associated to a submitted process pool transfer request

        :type monitor: TransferMonitor
        :param monitor: The monitor associated to the proccess pool downloader

        :type meta: ProcessPoolTransferMeta
        :param meta: The metadata associated to the request. This object
            is visible to the requester.
        """
        self._monitor = monitor
        self._meta = meta

    @property
    def meta(self):
        return self._meta

    def done(self):
        return self._monitor.is_done(self._meta.transfer_id)

    def result(self):
        try:
            return self._monitor.poll_for_result(self._meta.transfer_id)
        except KeyboardInterrupt:
            self._monitor._connect()
            self.cancel()
            raise

    def cancel(self):
        self._monitor.notify_exception(self._meta.transfer_id, CancelledError())