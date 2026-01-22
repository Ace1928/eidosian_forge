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
def _get_download_output_manager_cls(self, transfer_future, osutil):
    """Retrieves a class for managing output for a download

        :type transfer_future: s3transfer.futures.TransferFuture
        :param transfer_future: The transfer future for the request

        :type osutil: s3transfer.utils.OSUtils
        :param osutil: The os utility associated to the transfer

        :rtype: class of DownloadOutputManager
        :returns: The appropriate class to use for managing a specific type of
            input for downloads.
        """
    download_manager_resolver_chain = [DownloadSpecialFilenameOutputManager, DownloadFilenameOutputManager, DownloadSeekableOutputManager, DownloadNonSeekableOutputManager]
    fileobj = transfer_future.meta.call_args.fileobj
    for download_manager_cls in download_manager_resolver_chain:
        if download_manager_cls.is_compatible(fileobj, osutil):
            return download_manager_cls
    raise RuntimeError('Output %s of type: %s is not supported.' % (fileobj, type(fileobj)))