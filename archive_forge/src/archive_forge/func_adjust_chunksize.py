import random
import time
import functools
import math
import os
import socket
import stat
import string
import logging
import threading
import io
from collections import defaultdict
from botocore.exceptions import IncompleteReadError
from botocore.exceptions import ReadTimeoutError
from s3transfer.compat import SOCKET_ERROR
from s3transfer.compat import rename_file
from s3transfer.compat import seekable
from s3transfer.compat import fallocate
def adjust_chunksize(self, current_chunksize, file_size=None):
    """Get a chunksize close to current that fits within all S3 limits.

        :type current_chunksize: int
        :param current_chunksize: The currently configured chunksize.

        :type file_size: int or None
        :param file_size: The size of the file to upload. This might be None
            if the object being transferred has an unknown size.

        :returns: A valid chunksize that fits within configured limits.
        """
    chunksize = current_chunksize
    if file_size is not None:
        chunksize = self._adjust_for_max_parts(chunksize, file_size)
    return self._adjust_for_chunksize_limits(chunksize)