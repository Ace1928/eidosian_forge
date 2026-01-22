import copy
import logging
import threading
from botocore.compat import six
from s3transfer.constants import KB, MB
from s3transfer.constants import ALLOWED_DOWNLOAD_ARGS
from s3transfer.utils import get_callbacks
from s3transfer.utils import signal_transferring
from s3transfer.utils import signal_not_transferring
from s3transfer.utils import CallArgs
from s3transfer.utils import OSUtils
from s3transfer.utils import TaskSemaphore
from s3transfer.utils import SlidingWindowSemaphore
from s3transfer.exceptions import CancelledError
from s3transfer.exceptions import FatalError
from s3transfer.futures import IN_MEMORY_DOWNLOAD_TAG
from s3transfer.futures import IN_MEMORY_UPLOAD_TAG
from s3transfer.futures import BoundedExecutor
from s3transfer.futures import TransferFuture
from s3transfer.futures import TransferMeta
from s3transfer.futures import TransferCoordinator
from s3transfer.download import DownloadSubmissionTask
from s3transfer.upload import UploadSubmissionTask
from s3transfer.copies import CopySubmissionTask
from s3transfer.delete import DeleteSubmissionTask
from s3transfer.bandwidth import LeakyBucket
from s3transfer.bandwidth import BandwidthLimiter
def _submit_transfer(self, call_args, submission_task_cls, extra_main_kwargs=None):
    if not extra_main_kwargs:
        extra_main_kwargs = {}
    transfer_future, components = self._get_future_with_components(call_args)
    for callback in get_callbacks(transfer_future, 'done'):
        components['coordinator'].add_done_callback(callback)
    main_kwargs = self._get_submission_task_main_kwargs(transfer_future, extra_main_kwargs)
    self._submission_executor.submit(submission_task_cls(transfer_coordinator=components['coordinator'], main_kwargs=main_kwargs))
    self._id_counter += 1
    return transfer_future