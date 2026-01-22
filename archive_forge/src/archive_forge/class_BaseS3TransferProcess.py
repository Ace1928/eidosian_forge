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
class BaseS3TransferProcess(multiprocessing.Process):

    def __init__(self, client_factory):
        super(BaseS3TransferProcess, self).__init__()
        self._client_factory = client_factory
        self._client = None

    def run(self):
        self._client = self._client_factory.create_client()
        with ignore_ctrl_c():
            self._do_run()

    def _do_run(self):
        raise NotImplementedError('_do_run()')