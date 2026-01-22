import os
import math
import threading
import hashlib
import time
import logging
from boto.compat import Queue
import binascii
from boto.glacier.utils import DEFAULT_PART_SIZE, minimum_part_size, \
from boto.glacier.exceptions import UploadArchiveError, \
class DownloadWorkerThread(TransferThread):

    def __init__(self, job, worker_queue, result_queue, num_retries=5, time_between_retries=5, retry_exceptions=Exception):
        """
        Individual download thread that will download parts of the file from Glacier. Parts
        to download stored in work queue.

        Parts download to a temp dir with each part a separate file

        :param job: Glacier job object
        :param work_queue: A queue of tuples which include the part_number and
            part_size
        :param result_queue: A priority queue of tuples which include the
            part_number and the path to the temp file that holds that
            part's data.

        """
        super(DownloadWorkerThread, self).__init__(worker_queue, result_queue)
        self._job = job
        self._num_retries = num_retries
        self._time_between_retries = time_between_retries
        self._retry_exceptions = retry_exceptions

    def _process_chunk(self, work):
        """
        Attempt to download a part of the archive from Glacier
        Store the result in the result_queue

        :param work:
        """
        result = None
        for _ in range(self._num_retries):
            try:
                result = self._download_chunk(work)
                break
            except self._retry_exceptions as e:
                log.error('Exception caught downloading part number %s for job %s', work[0], self._job)
                time.sleep(self._time_between_retries)
                result = e
        return result

    def _download_chunk(self, work):
        """
        Downloads a chunk of archive from Glacier. Saves the data to a temp file
        Returns the part number and temp file location

        :param work:
        """
        part_number, part_size = work
        start_byte = part_number * part_size
        byte_range = (start_byte, start_byte + part_size - 1)
        log.debug('Downloading chunk %s of size %s', part_number, part_size)
        response = self._job.get_output(byte_range)
        data = response.read()
        actual_hash = bytes_to_hex(tree_hash(chunk_hashes(data)))
        if response['TreeHash'] != actual_hash:
            raise TreeHashDoesNotMatchError('Tree hash for part number %s does not match, expected: %s, got: %s' % (part_number, response['TreeHash'], actual_hash))
        return (part_number, part_size, binascii.unhexlify(actual_hash), data)