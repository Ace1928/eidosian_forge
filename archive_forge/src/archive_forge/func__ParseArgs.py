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
def _ParseArgs(self):
    """Parses arguments for perfdiag command."""
    self.num_objects = 5
    self.processes = 1
    self.threads = 1
    self.parallel_strategy = None
    self.num_slices = 4
    self.thru_filesize = 1048576
    self.directory = tempfile.gettempdir()
    self.delete_directory = False
    self.diag_tests = set(self.DEFAULT_DIAG_TESTS)
    self.output_file = None
    self.input_file = None
    self.metadata_keys = {}
    self.gzip_encoded_writes = False
    self.gzip_compression_ratio = 100
    if self.sub_opts:
        for o, a in self.sub_opts:
            if o == '-n':
                self.num_objects = self._ParsePositiveInteger(a, 'The -n parameter must be a positive integer.')
            if o == '-c':
                self.processes = self._ParsePositiveInteger(a, 'The -c parameter must be a positive integer.')
            if o == '-k':
                self.threads = self._ParsePositiveInteger(a, 'The -k parameter must be a positive integer.')
            if o == '-p':
                if a.lower() in self.PARALLEL_STRATEGIES:
                    self.parallel_strategy = a.lower()
                else:
                    raise CommandException("'%s' is not a valid parallelism strategy." % a)
            if o == '-y':
                self.num_slices = self._ParsePositiveInteger(a, 'The -y parameter must be a positive integer.')
            if o == '-s':
                try:
                    self.thru_filesize = HumanReadableToBytes(a)
                except ValueError:
                    raise CommandException('Invalid -s parameter.')
            if o == '-d':
                self.directory = a
                if not os.path.exists(self.directory):
                    self.delete_directory = True
                    os.makedirs(self.directory)
            if o == '-t':
                self.diag_tests = set()
                for test_name in a.strip().split(','):
                    if test_name.lower() not in self.ALL_DIAG_TESTS:
                        raise CommandException("List of test names (-t) contains invalid test name '%s'." % test_name)
                    self.diag_tests.add(test_name)
            if o == '-m':
                pieces = a.split(':')
                if len(pieces) != 2:
                    raise CommandException("Invalid metadata key-value combination '%s'." % a)
                key, value = pieces
                self.metadata_keys[key] = value
            if o == '-o':
                self.output_file = os.path.abspath(a)
            if o == '-i':
                self.input_file = os.path.abspath(a)
                if not os.path.isfile(self.input_file):
                    raise CommandException("Invalid input file (-i): '%s'." % a)
                try:
                    with open(self.input_file, 'r') as f:
                        self.results = json.load(f)
                        self.logger.info("Read input file: '%s'.", self.input_file)
                except ValueError:
                    raise CommandException("Could not decode input file (-i): '%s'." % a)
                return
            if o == '-j':
                self.gzip_encoded_writes = True
                try:
                    self.gzip_compression_ratio = int(a)
                except ValueError:
                    self.gzip_compression_ratio = -1
                if self.gzip_compression_ratio < 0 or self.gzip_compression_ratio > 100:
                    raise CommandException('The -j parameter must be between 0 and 100 (inclusive).')
    if (self.processes > 1 or self.threads > 1) and (not self.parallel_strategy):
        self.parallel_strategy = self.FAN
    elif self.processes == 1 and self.threads == 1 and self.parallel_strategy:
        raise CommandException('Cannot specify parallelism strategy (-p) without also specifying multiple threads and/or processes (-c and/or -k).')
    if not self.args:
        self.RaiseWrongNumberOfArgumentsException()
    self.bucket_url = StorageUrlFromString(self.args[0])
    self.provider = self.bucket_url.scheme
    if not self.bucket_url.IsCloudUrl() and self.bucket_url.IsBucket():
        raise CommandException('The perfdiag command requires a URL that specifies a bucket.\n"%s" is not valid.' % self.args[0])
    if self.thru_filesize > HumanReadableToBytes('2GiB') and (self.RTHRU in self.diag_tests or self.WTHRU in self.diag_tests):
        raise CommandException('For in-memory tests maximum file size is 2GiB. For larger file sizes, specify rthru_file and/or wthru_file with the -t option.')
    perform_slice = self.parallel_strategy in (self.SLICE, self.BOTH)
    slice_not_available = self.provider == 's3' and self.diag_tests.intersection(self.WTHRU, self.WTHRU_FILE)
    if perform_slice and slice_not_available:
        raise CommandException('Sliced uploads are not available for s3. Use -p fan or sequential uploads for s3.')
    self.gsutil_api.GetBucket(self.bucket_url.bucket_name, provider=self.bucket_url.scheme, fields=['id'])
    self.exceptions = [http_client.HTTPException, socket.error, socket.gaierror, socket.timeout, http_client.BadStatusLine, ServiceException]