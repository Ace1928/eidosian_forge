from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import shutil
from gslib import storage_url
from gslib.utils import execution_util
from gslib.utils import temporary_file_util
from boto import config
STET-decrypts downloaded file.

  Args:
    source_url (StorageUrl): Copy source.
    destination_url (StorageUrl): Copy destination.
    temporary_file_name (str): Path to temporary file used for download.
    logger (logging.Logger): For logging STET binary output.
  