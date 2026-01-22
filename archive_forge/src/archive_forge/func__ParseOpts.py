from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import errno
import heapq
import io
from itertools import islice
import logging
import os
import re
import tempfile
import textwrap
import time
import traceback
import sys
import six
from six.moves import urllib
from boto import config
import crcmod
from gslib.bucket_listing_ref import BucketListingObject
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import ServiceException
from gslib.command import Command
from gslib.command import DummyArgChecker
from gslib.commands.cp import ShimTranslatePredefinedAclSubOptForCopy
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.metrics import LogPerformanceSummaryParams
from gslib.plurality_checkable_iterator import PluralityCheckableIterator
from gslib.seek_ahead_thread import SeekAheadResult
from gslib.sig_handling import GetCaughtSignals
from gslib.sig_handling import RegisterSignalHandler
from gslib.storage_url import GenerationFromUrlAndString
from gslib.storage_url import IsCloudSubdirPlaceholder
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import constants
from gslib.utils import copy_helper
from gslib.utils import parallelism_framework_util
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.copy_helper import CreateCopyHelperOpts
from gslib.utils.copy_helper import GetSourceFieldsNeededForCopy
from gslib.utils.copy_helper import GZIP_ALL_FILES
from gslib.utils.copy_helper import SkipUnsupportedObjectError
from gslib.utils.hashing_helper import CalculateB64EncodedCrc32cFromContents
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.hashing_helper import SLOW_CRCMOD_RSYNC_WARNING
from gslib.utils.hashing_helper import SLOW_CRCMOD_WARNING
from gslib.utils.metadata_util import CreateCustomMetadata
from gslib.utils.metadata_util import GetValueFromObjectCustomMetadata
from gslib.utils.metadata_util import ObjectIsGzipEncoded
from gslib.utils.posix_util import ATIME_ATTR
from gslib.utils.posix_util import ConvertDatetimeToPOSIX
from gslib.utils.posix_util import ConvertModeToBase8
from gslib.utils.posix_util import DeserializeFileAttributesFromObjectMetadata
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import InitializePreservePosixData
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import NA_ID
from gslib.utils.posix_util import NA_MODE
from gslib.utils.posix_util import NA_TIME
from gslib.utils.posix_util import NeedsPOSIXAttributeUpdate
from gslib.utils.posix_util import ParseAndSetPOSIXAttributes
from gslib.utils.posix_util import POSIXAttributes
from gslib.utils.posix_util import SerializeFileAttributesToObjectMetadata
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.posix_util import ValidateFilePermissionAccess
from gslib.utils.posix_util import WarnFutureTimestamp
from gslib.utils.posix_util import WarnInvalidValue
from gslib.utils.posix_util import WarnNegativeAttribute
from gslib.utils.rsync_util import DiffAction
from gslib.utils.rsync_util import RsyncDiffToApply
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.translation_helper import CopyCustomMetadata
from gslib.utils.unit_util import CalculateThroughput
from gslib.utils.unit_util import SECONDS_PER_DAY
from gslib.utils.unit_util import TEN_MIB
from gslib.wildcard_iterator import CreateWildcardIterator
def _ParseOpts(self):
    self.exclude_symlinks = False
    self.continue_on_error = False
    self.delete_extras = False
    self.preserve_acl = False
    self.preserve_posix_attrs = False
    self.compute_file_checksums = False
    self.dryrun = False
    self.exclude_dirs = False
    self.exclude_pattern = None
    self.skip_old_files = False
    self.ignore_existing = False
    self.skip_unsupported_objects = False
    canned_acl = None
    self.canned = None
    gzip_encoded = False
    gzip_arg_exts = None
    gzip_arg_all = None
    if self.sub_opts:
        for o, a in self.sub_opts:
            if o == '-a':
                canned_acl = a
                self.canned = True
            if o == '-c':
                self.compute_file_checksums = True
            elif o == '-C':
                self.continue_on_error = True
            elif o == '-d':
                self.delete_extras = True
            elif o == '-e':
                self.exclude_symlinks = True
            elif o == '-j':
                gzip_encoded = True
                gzip_arg_exts = [x.strip() for x in a.split(',')]
            elif o == '-J':
                gzip_encoded = True
                gzip_arg_all = GZIP_ALL_FILES
            elif o == '-n':
                self.dryrun = True
            elif o == '-p':
                self.preserve_acl = True
            elif o == '-P':
                self.preserve_posix_attrs = True
                if not IS_WINDOWS:
                    InitializePreservePosixData()
            elif o == '-r' or o == '-R':
                self.recursion_requested = True
            elif o == '-u':
                self.skip_old_files = True
            elif o == '-i':
                self.ignore_existing = True
            elif o == '-U':
                self.skip_unsupported_objects = True
            elif o == '-x' or o == '-y':
                if o == '-y':
                    self.exclude_dirs = True
                if not a:
                    raise CommandException('Invalid blank exclude filter')
                try:
                    self.exclude_pattern = re.compile(a)
                except re.error:
                    raise CommandException('Invalid exclude filter (%s)' % a)
    if self.preserve_acl and canned_acl:
        raise CommandException('Specifying both the -p and -a options together is invalid.')
    if gzip_arg_exts and gzip_arg_all:
        raise CommandException('Specifying both the -j and -J options together is invalid.')
    self.gzip_encoded = gzip_encoded
    self.gzip_exts = gzip_arg_exts or gzip_arg_all
    return CreateCopyHelperOpts(canned_acl=canned_acl, preserve_acl=self.preserve_acl, skip_unsupported_objects=self.skip_unsupported_objects)