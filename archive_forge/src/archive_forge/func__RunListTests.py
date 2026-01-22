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
def _RunListTests(self):
    """Runs eventual consistency listing latency tests."""
    self.results['listing'] = {'num_files': self.num_objects}
    list_objects = []
    args = []
    random_id = ''.join([random.choice(string.ascii_lowercase) for _ in range(10)])
    list_prefix = 'gsutil-perfdiag-list-' + random_id + '-'
    for _ in xrange(self.num_objects):
        fpath = self._MakeTempFile(0, mem_data=True, mem_metadata=True, prefix=list_prefix)
        object_name = os.path.basename(fpath)
        list_objects.append(object_name)
        args.append(FanUploadTuple(False, fpath, object_name, False, False))
        self.temporary_objects.add(object_name)
    self.logger.info('\nWriting %s objects for listing test...', self.num_objects)
    self.Apply(_UploadObject, args, _PerfdiagExceptionHandler, arg_checker=DummyArgChecker)
    list_latencies = []
    files_seen = []
    total_start_time = time.time()
    expected_objects = set(list_objects)
    found_objects = set()

    def _List():
        """Lists and returns objects in the bucket. Also records latency."""
        t0 = time.time()
        objects = list(self.gsutil_api.ListObjects(self.bucket_url.bucket_name, prefix=list_prefix, delimiter='/', provider=self.provider, fields=['items/name']))
        if len(objects) > self.num_objects:
            self.logger.warning('Listing produced more than the expected %d object(s).', self.num_objects)
        t1 = time.time()
        list_latencies.append(t1 - t0)
        return set([obj.data.name for obj in objects])

    def _ListAfterUpload():
        names = _List()
        found_objects.update(names & expected_objects)
        files_seen.append(len(found_objects))
    self.logger.info('Listing bucket %s waiting for %s objects to appear...', self.bucket_url.bucket_name, self.num_objects)
    while expected_objects - found_objects:
        self._RunOperation(_ListAfterUpload)
        if expected_objects - found_objects:
            if time.time() - total_start_time > self.MAX_LISTING_WAIT_TIME:
                self.logger.warning('Maximum time reached waiting for listing.')
                break
    total_end_time = time.time()
    self.results['listing']['insert'] = {'num_listing_calls': len(list_latencies), 'list_latencies': list_latencies, 'files_seen_after_listing': files_seen, 'time_took': total_end_time - total_start_time}
    args = [object_name for object_name in list_objects]
    self.logger.info('Deleting %s objects for listing test...', self.num_objects)
    self.Apply(_DeleteWrapper, args, _PerfdiagExceptionHandler, arg_checker=DummyArgChecker)
    self.logger.info('Listing bucket %s waiting for %s objects to disappear...', self.bucket_url.bucket_name, self.num_objects)
    list_latencies = []
    files_seen = []
    total_start_time = time.time()
    found_objects = set(list_objects)
    while found_objects:

        def _ListAfterDelete():
            names = _List()
            found_objects.intersection_update(names)
            files_seen.append(len(found_objects))
        self._RunOperation(_ListAfterDelete)
        if found_objects:
            if time.time() - total_start_time > self.MAX_LISTING_WAIT_TIME:
                self.logger.warning('Maximum time reached waiting for listing.')
                break
    total_end_time = time.time()
    self.results['listing']['delete'] = {'num_listing_calls': len(list_latencies), 'list_latencies': list_latencies, 'files_seen_after_listing': files_seen, 'time_took': total_end_time - total_start_time}