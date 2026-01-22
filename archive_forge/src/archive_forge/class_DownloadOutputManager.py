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
class DownloadOutputManager(object):
    """Base manager class for handling various types of files for downloads

    This class is typically used for the DownloadSubmissionTask class to help
    determine the following:

        * Provides the fileobj to write to downloads to
        * Get a task to complete once everything downloaded has been written

    The answers/implementations differ for the various types of file outputs
    that may be accepted. All implementations must subclass and override
    public methods from this class.
    """

    def __init__(self, osutil, transfer_coordinator, io_executor):
        self._osutil = osutil
        self._transfer_coordinator = transfer_coordinator
        self._io_executor = io_executor

    @classmethod
    def is_compatible(cls, download_target, osutil):
        """Determines if the target for the download is compatible with manager

        :param download_target: The target for which the upload will write
            data to.

        :param osutil: The os utility to be used for the transfer

        :returns: True if the manager can handle the type of target specified
            otherwise returns False.
        """
        raise NotImplementedError('must implement is_compatible()')

    def get_download_task_tag(self):
        """Get the tag (if any) to associate all GetObjectTasks

        :rtype: s3transfer.futures.TaskTag
        :returns: The tag to associate all GetObjectTasks with
        """
        return None

    def get_fileobj_for_io_writes(self, transfer_future):
        """Get file-like object to use for io writes in the io executor

        :type transfer_future: s3transfer.futures.TransferFuture
        :param transfer_future: The future associated with upload request

        returns: A file-like object to write to
        """
        raise NotImplementedError('must implement get_fileobj_for_io_writes()')

    def queue_file_io_task(self, fileobj, data, offset):
        """Queue IO write for submission to the IO executor.

        This method accepts an IO executor and information about the
        downloaded data, and handles submitting this to the IO executor.

        This method may defer submission to the IO executor if necessary.

        """
        self._transfer_coordinator.submit(self._io_executor, self.get_io_write_task(fileobj, data, offset))

    def get_io_write_task(self, fileobj, data, offset):
        """Get an IO write task for the requested set of data

        This task can be ran immediately or be submitted to the IO executor
        for it to run.

        :type fileobj: file-like object
        :param fileobj: The file-like object to write to

        :type data: bytes
        :param data: The data to write out

        :type offset: integer
        :param offset: The offset to write the data to in the file-like object

        :returns: An IO task to be used to write data to a file-like object
        """
        return IOWriteTask(self._transfer_coordinator, main_kwargs={'fileobj': fileobj, 'data': data, 'offset': offset})

    def get_final_io_task(self):
        """Get the final io task to complete the download

        This is needed because based on the architecture of the TransferManager
        the final tasks will be sent to the IO executor, but the executor
        needs a final task for it to signal that the transfer is done and
        all done callbacks can be run.

        :rtype: s3transfer.tasks.Task
        :returns: A final task to completed in the io executor
        """
        raise NotImplementedError('must implement get_final_io_task()')

    def _get_fileobj_from_filename(self, filename):
        f = DeferredOpenFile(filename, mode='wb', open_function=self._osutil.open)
        self._transfer_coordinator.add_failure_cleanup(f.close)
        return f