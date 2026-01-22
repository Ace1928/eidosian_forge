from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import time
import crcmod
import six
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.progress_callback import FileProgressCallbackHandler
from gslib.progress_callback import ProgressCallbackWithTimeout
from gslib.storage_url import StorageUrlFromString
from gslib.thread_message import FileMessage
from gslib.thread_message import FinalMessage
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import hashing_helper
from gslib.utils import parallelism_framework_util
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils import shim_util
def _GetHashClassesFromArgs(self, calc_crc32c, calc_md5):
    """Constructs the dictionary of hashes to compute based on the arguments.

    Args:
      calc_crc32c: If True, CRC32c should be included.
      calc_md5: If True, MD5 should be included.

    Returns:
      Dictionary of {string: hash digester}, where string the name of the
          digester algorithm.
    """
    hash_dict = {}
    if calc_crc32c:
        hash_dict['crc32c'] = crcmod.predefined.Crc('crc-32c')
    if calc_md5:
        hash_dict['md5'] = hashing_helper.GetMd5()
    return hash_dict