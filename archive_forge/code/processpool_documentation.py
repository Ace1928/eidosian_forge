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
Fulfills GetObjectJobs

        Downloads the S3 object, writes it to the specified file, and
        renames the file to its final location if it completes the final
        job for a particular transfer.

        :param queue: Queue for retrieving GetObjectJob's
        :param client_factory: ClientFactory for creating S3 clients
        :param transfer_monitor: Monitor for notifying
        :param osutil: OSUtils object to use for os-related behavior when
            performing the transfer.
        