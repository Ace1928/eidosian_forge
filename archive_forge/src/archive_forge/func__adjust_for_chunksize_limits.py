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
def _adjust_for_chunksize_limits(self, current_chunksize):
    if current_chunksize > self.max_size:
        logger.debug('Chunksize greater than maximum chunksize. Setting to %s from %s.' % (self.max_size, current_chunksize))
        return self.max_size
    elif current_chunksize < self.min_size:
        logger.debug('Chunksize less than minimum chunksize. Setting to %s from %s.' % (self.min_size, current_chunksize))
        return self.min_size
    else:
        return current_chunksize