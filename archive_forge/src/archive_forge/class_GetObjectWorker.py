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
class GetObjectWorker(BaseS3TransferProcess):
    _MAX_ATTEMPTS = 5
    _IO_CHUNKSIZE = 2 * MB

    def __init__(self, queue, client_factory, transfer_monitor, osutil):
        """Fulfills GetObjectJobs

        Downloads the S3 object, writes it to the specified file, and
        renames the file to its final location if it completes the final
        job for a particular transfer.

        :param queue: Queue for retrieving GetObjectJob's
        :param client_factory: ClientFactory for creating S3 clients
        :param transfer_monitor: Monitor for notifying
        :param osutil: OSUtils object to use for os-related behavior when
            performing the transfer.
        """
        super(GetObjectWorker, self).__init__(client_factory)
        self._queue = queue
        self._client_factory = client_factory
        self._transfer_monitor = transfer_monitor
        self._osutil = osutil

    def _do_run(self):
        while True:
            job = self._queue.get()
            if job == SHUTDOWN_SIGNAL:
                logger.debug('Worker shutdown signal received.')
                return
            if not self._transfer_monitor.get_exception(job.transfer_id):
                self._run_get_object_job(job)
            else:
                logger.debug('Skipping get object job %s because there was a previous exception.', job)
            remaining = self._transfer_monitor.notify_job_complete(job.transfer_id)
            logger.debug('%s jobs remaining for transfer_id %s.', remaining, job.transfer_id)
            if not remaining:
                self._finalize_download(job.transfer_id, job.temp_filename, job.filename)

    def _run_get_object_job(self, job):
        try:
            self._do_get_object(bucket=job.bucket, key=job.key, temp_filename=job.temp_filename, extra_args=job.extra_args, offset=job.offset)
        except Exception as e:
            logger.debug('Exception caught when downloading object for get object job %s: %s', job, e, exc_info=True)
            self._transfer_monitor.notify_exception(job.transfer_id, e)

    def _do_get_object(self, bucket, key, extra_args, temp_filename, offset):
        last_exception = None
        for i in range(self._MAX_ATTEMPTS):
            try:
                response = self._client.get_object(Bucket=bucket, Key=key, **extra_args)
                self._write_to_file(temp_filename, offset, response['Body'])
                return
            except S3_RETRYABLE_DOWNLOAD_ERRORS as e:
                logger.debug('Retrying exception caught (%s), retrying request, (attempt %s / %s)', e, i + 1, self._MAX_ATTEMPTS, exc_info=True)
                last_exception = e
        raise RetriesExceededError(last_exception)

    def _write_to_file(self, filename, offset, body):
        with open(filename, 'rb+') as f:
            f.seek(offset)
            chunks = iter(lambda: body.read(self._IO_CHUNKSIZE), b'')
            for chunk in chunks:
                f.write(chunk)

    def _finalize_download(self, transfer_id, temp_filename, filename):
        if self._transfer_monitor.get_exception(transfer_id):
            self._osutil.remove_file(temp_filename)
        else:
            self._do_file_rename(transfer_id, temp_filename, filename)
        self._transfer_monitor.notify_done(transfer_id)

    def _do_file_rename(self, transfer_id, temp_filename, filename):
        try:
            self._osutil.rename_file(temp_filename, filename)
        except Exception as e:
            self._transfer_monitor.notify_exception(transfer_id, e)
            self._osutil.remove_file(temp_filename)