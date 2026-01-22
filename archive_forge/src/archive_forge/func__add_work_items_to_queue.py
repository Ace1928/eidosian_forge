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
def _add_work_items_to_queue(self, total_parts, worker_queue, part_size):
    log.debug('Adding work items to queue.')
    for i in range(total_parts):
        worker_queue.put((i, part_size))
    for i in range(self._num_threads):
        worker_queue.put(_END_SENTINEL)