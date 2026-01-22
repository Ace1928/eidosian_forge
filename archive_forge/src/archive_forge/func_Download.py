from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import calendar
from collections import defaultdict
from collections import namedtuple
import contextlib
import datetime
import json
import logging
import math
import multiprocessing
import os
import random
import re
import socket
import string
import subprocess
import tempfile
import time
import boto
import boto.gs.connection
import six
from six.moves import cStringIO
from six.moves import http_client
from six.moves import xrange
from six.moves import range
import gslib
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import DummyArgChecker
from gslib.command_argument import CommandArgument
from gslib.commands import config
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.file_part import FilePart
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import text_util
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import ResumableThreshold
from gslib.utils.constants import UTF8
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.cloud_api_helper import GetDownloadSerializationData
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.system_util import CheckFreeSpace
from gslib.utils.system_util import GetDiskCounters
from gslib.utils.system_util import GetFileSize
from gslib.utils.system_util import IS_LINUX
from gslib.utils.system_util import IsRunningInCiEnvironment
from gslib.utils.unit_util import DivideAndCeil
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import MakeBitsHumanReadable
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import Percentile
def Download(self, object_name, gsutil_api, file_name=None, serialization_data=None, start_byte=0, end_byte=None):
    """Downloads an object from the test bucket.

    Args:
      object_name: The name of the object (in the test bucket) to download.
      gsutil_api: CloudApi instance to use for the download.
      file_name: Optional file name to write downloaded data to. If None,
                 downloaded data is discarded immediately.
      serialization_data: Optional serialization data, used so that we don't
                          have to get the metadata before downloading.
      start_byte: The first byte in the object to download.
                  (only should be specified for sliced downloads)
      end_byte: The last byte in the object to download.
                (only should be specified for sliced downloads)
    """
    fp = None
    try:
        if file_name is not None:
            fp = open(file_name, 'r+b')
            fp.seek(start_byte)
        else:
            fp = self.discard_sink

        def _InnerDownload():
            gsutil_api.GetObjectMedia(self.bucket_url.bucket_name, object_name, fp, provider=self.provider, start_byte=start_byte, end_byte=end_byte, serialization_data=serialization_data)
        self._RunOperation(_InnerDownload)
    finally:
        if fp:
            fp.close()