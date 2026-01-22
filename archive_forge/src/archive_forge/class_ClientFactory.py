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
class ClientFactory(object):

    def __init__(self, client_kwargs=None):
        """Creates S3 clients for processes

        Botocore sessions and clients are not pickleable so they cannot be
        inherited across Process boundaries. Instead, they must be instantiated
        once a process is running.
        """
        self._client_kwargs = client_kwargs
        if self._client_kwargs is None:
            self._client_kwargs = {}
        client_config = deepcopy(self._client_kwargs.get('config', Config()))
        if not client_config.user_agent_extra:
            client_config.user_agent_extra = PROCESS_USER_AGENT
        else:
            client_config.user_agent_extra += ' ' + PROCESS_USER_AGENT
        self._client_kwargs['config'] = client_config

    def create_client(self):
        """Create a botocore S3 client"""
        return botocore.session.Session().create_client('s3', **self._client_kwargs)