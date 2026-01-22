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
def _RunLatencyTests(self):
    """Runs latency tests."""
    self.results['latency'] = defaultdict(list)
    for i in range(self.num_objects):
        self.logger.info('\nRunning latency iteration %d...', i + 1)
        for fpath in self.latency_files:
            file_data = temp_file_dict[fpath]
            url = self.bucket_url.Clone()
            url.object_name = os.path.basename(fpath)
            file_size = file_data.size
            readable_file_size = MakeHumanReadable(file_size)
            self.logger.info("\nFile of size %s located on disk at '%s' being diagnosed in the cloud at '%s'.", readable_file_size, fpath, url)
            upload_target = StorageUrlToUploadObjectMetadata(url)

            def _Upload():
                io_fp = six.BytesIO(file_data.data)
                with self._Time('UPLOAD_%d' % file_size, self.results['latency']):
                    self.gsutil_api.UploadObject(io_fp, upload_target, size=file_size, provider=self.provider, fields=['name'])
            self._RunOperation(_Upload)

            def _Metadata():
                with self._Time('METADATA_%d' % file_size, self.results['latency']):
                    return self.gsutil_api.GetObjectMetadata(url.bucket_name, url.object_name, provider=self.provider, fields=['name', 'contentType', 'mediaLink', 'size'])
            download_metadata = self._RunOperation(_Metadata)
            serialization_data = GetDownloadSerializationData(download_metadata)

            def _Download():
                with self._Time('DOWNLOAD_%d' % file_size, self.results['latency']):
                    self.gsutil_api.GetObjectMedia(url.bucket_name, url.object_name, self.discard_sink, provider=self.provider, serialization_data=serialization_data)
            self._RunOperation(_Download)

            def _Delete():
                with self._Time('DELETE_%d' % file_size, self.results['latency']):
                    self.gsutil_api.DeleteObject(url.bucket_name, url.object_name, provider=self.provider)
            self._RunOperation(_Delete)