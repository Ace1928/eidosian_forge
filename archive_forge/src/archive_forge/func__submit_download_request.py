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
def _submit_download_request(self, client, config, osutil, request_executor, io_executor, download_output_manager, transfer_future, bandwidth_limiter):
    call_args = transfer_future.meta.call_args
    fileobj = download_output_manager.get_fileobj_for_io_writes(transfer_future)
    progress_callbacks = get_callbacks(transfer_future, 'progress')
    get_object_tag = download_output_manager.get_download_task_tag()
    final_task = download_output_manager.get_final_io_task()
    self._transfer_coordinator.submit(request_executor, ImmediatelyWriteIOGetObjectTask(transfer_coordinator=self._transfer_coordinator, main_kwargs={'client': client, 'bucket': call_args.bucket, 'key': call_args.key, 'fileobj': fileobj, 'extra_args': call_args.extra_args, 'callbacks': progress_callbacks, 'max_attempts': config.num_download_attempts, 'download_output_manager': download_output_manager, 'io_chunksize': config.io_chunksize, 'bandwidth_limiter': bandwidth_limiter}, done_callbacks=[final_task]), tag=get_object_tag)