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
class _DiffIterator(object):
    """Iterator yielding sequence of RsyncDiffToApply objects."""

    def __init__(self, command_obj, base_src_url, base_dst_url):
        self.command_obj = command_obj
        self.compute_file_checksums = command_obj.compute_file_checksums
        self.delete_extras = command_obj.delete_extras
        self.recursion_requested = command_obj.recursion_requested
        self.logger = self.command_obj.logger
        self.base_src_url = base_src_url
        self.base_dst_url = base_dst_url
        self.preserve_posix = command_obj.preserve_posix_attrs
        self.skip_old_files = command_obj.skip_old_files
        self.ignore_existing = command_obj.ignore_existing
        self.logger.info('Building synchronization state...')
        temp_src_file = tempfile.NamedTemporaryFile(prefix='gsutil-rsync-src-', delete=False)
        temp_dst_file = tempfile.NamedTemporaryFile(prefix='gsutil-rsync-dst-', delete=False)
        self.sorted_list_src_file_name = temp_src_file.name
        self.sorted_list_dst_file_name = temp_dst_file.name
        _tmp_files.append(temp_src_file)
        _tmp_files.append(temp_dst_file)
        temp_src_file.close()
        temp_dst_file.close()
        args_iter = iter([(self.base_src_url.url_string, self.sorted_list_src_file_name, 'source'), (self.base_dst_url.url_string, self.sorted_list_dst_file_name, 'destination')])
        command_obj.non_retryable_listing_failures = 0
        shared_attrs = ['non_retryable_listing_failures']
        command_obj.Apply(_ListUrlRootFunc, args_iter, _RootListingExceptionHandler, shared_attrs, arg_checker=DummyArgChecker, parallel_operations_override=command_obj.ParallelOverrideReason.SPEED, fail_on_error=True)
        if command_obj.non_retryable_listing_failures:
            raise CommandException('Caught non-retryable exception - aborting rsync')
        self.sorted_list_src_file = open(self.sorted_list_src_file_name, 'r')
        self.sorted_list_dst_file = open(self.sorted_list_dst_file_name, 'r')
        _tmp_files.append(self.sorted_list_src_file)
        _tmp_files.append(self.sorted_list_dst_file)
        if base_src_url.IsCloudUrl() and base_dst_url.IsFileUrl() and self.preserve_posix:
            self.sorted_src_urls_it = PluralityCheckableIterator(iter(self.sorted_list_src_file))
            self._ValidateObjectAccess()
            self.sorted_list_src_file.seek(0)
        self.sorted_src_urls_it = PluralityCheckableIterator(iter(self.sorted_list_src_file))
        self.sorted_dst_urls_it = PluralityCheckableIterator(iter(self.sorted_list_dst_file))

    def _ValidateObjectAccess(self):
        """Validates that the user won't lose access to the files if copied.

    Iterates over the src file list to check if access will be maintained. If at
    any point we would orphan a file, a list of errors is compiled and logged
    with an exception raised to the user.
    """
        errors = collections.deque()
        for src_url in self.sorted_src_urls_it:
            src_url_str, _, _, _, _, src_mode, src_uid, src_gid, _, _ = self._ParseTmpFileLine(src_url)
            valid, err = ValidateFilePermissionAccess(src_url_str, uid=src_uid, gid=src_gid, mode=src_mode)
            if not valid:
                errors.append(err)
        if errors:
            for err in errors:
                self.logger.critical(err)
            raise CommandException('This sync will orphan file(s), please fix their permissions before trying again.')

    def _ParseTmpFileLine(self, line):
        """Parses output from _BuildTmpOutputLine.

    Parses into tuple:
      (URL, size, time_created, atime, mtime, mode, uid, gid, crc32c, md5)
    where crc32c and/or md5 can be _NA and atime/mtime/time_created can be
    NA_TIME.

    Args:
      line: The line to parse.

    Returns:
      Parsed tuple: (url, size, time_created, atime, mtime, mode, uid, gid,
                     crc32c, md5)
    """
        encoded_url, size, time_created, atime, mtime, mode, uid, gid, crc32c, md5 = line.rsplit(None, 9)
        return (_DecodeUrl(encoded_url), int(size), long(time_created), long(atime), long(mtime), int(mode), int(uid), int(gid), crc32c, md5.strip())

    def _WarnIfMissingCloudHash(self, url_str, crc32c, md5):
        """Warns if given url_str is a cloud URL and is missing both crc32c and md5.

    Args:
      url_str: Destination URL string.
      crc32c: Destination CRC32c.
      md5: Destination MD5.

    Returns:
      True if issued warning.
    """
        if StorageUrlFromString(url_str).IsCloudUrl() and crc32c == _NA and (md5 == _NA):
            self.logger.warn('Found no hashes to validate %s. Integrity cannot be assured without hashes.', url_str)
            return True
        return False

    def _CompareObjects(self, src_url_str, src_size, src_mtime, src_crc32c, src_md5, dst_url_str, dst_size, dst_mtime, dst_crc32c, dst_md5):
        """Returns whether src should replace dst object, and if mtime is present.

    Uses mtime, size, or whatever checksums are available.

    Args:
      src_url_str: Source URL string.
      src_size: Source size.
      src_mtime: Source modification time.
      src_crc32c: Source CRC32c.
      src_md5: Source MD5.
      dst_url_str: Destination URL string.
      dst_size: Destination size.
      dst_mtime: Destination modification time.
      dst_crc32c: Destination CRC32c.
      dst_md5: Destination MD5.

    Returns:
      A 3-tuple indicating if src should replace dst, and if src and dst have
      mtime.
    """
        has_src_mtime = src_mtime > NA_TIME
        has_dst_mtime = dst_mtime > NA_TIME
        use_hashes = self.compute_file_checksums or (StorageUrlFromString(src_url_str).IsCloudUrl() and StorageUrlFromString(dst_url_str).IsCloudUrl())
        if self.ignore_existing:
            return (False, has_src_mtime, has_dst_mtime)
        if self.skip_old_files and has_src_mtime and has_dst_mtime and (src_mtime < dst_mtime):
            return (False, has_src_mtime, has_dst_mtime)
        if not use_hashes and has_src_mtime and has_dst_mtime:
            return (src_mtime != dst_mtime or src_size != dst_size, has_src_mtime, has_dst_mtime)
        if src_size != dst_size:
            return (True, has_src_mtime, has_dst_mtime)
        src_crc32c, src_md5, dst_crc32c, dst_md5 = _ComputeNeededFileChecksums(self.logger, src_url_str, src_size, src_crc32c, src_md5, dst_url_str, dst_size, dst_crc32c, dst_md5)
        if src_md5 != _NA and dst_md5 != _NA:
            self.logger.debug('Comparing md5 for %s and %s', src_url_str, dst_url_str)
            return (src_md5 != dst_md5, has_src_mtime, has_dst_mtime)
        if src_crc32c != _NA and dst_crc32c != _NA:
            self.logger.debug('Comparing crc32c for %s and %s', src_url_str, dst_url_str)
            return (src_crc32c != dst_crc32c, has_src_mtime, has_dst_mtime)
        if not self._WarnIfMissingCloudHash(src_url_str, src_crc32c, src_md5):
            self._WarnIfMissingCloudHash(dst_url_str, dst_crc32c, dst_md5)
        return (False, has_src_mtime, has_dst_mtime)

    def __iter__(self):
        """Iterates over src/dst URLs and produces a RsyncDiffToApply sequence.

    Yields:
      The RsyncDiffToApply.
    """
        base_src_url_len = len(self.base_src_url.url_string.rstrip('/\\'))
        base_dst_url_len = len(self.base_dst_url.url_string.rstrip('/\\'))
        out_of_src_items = False
        src_url_str = dst_url_str = None
        while True:
            if src_url_str is None:
                if self.sorted_src_urls_it.IsEmpty():
                    out_of_src_items = True
                else:
                    src_url_str, src_size, src_time_created, src_atime, src_mtime, src_mode, src_uid, src_gid, src_crc32c, src_md5 = self._ParseTmpFileLine(next(self.sorted_src_urls_it))
                    posix_attrs = POSIXAttributes(atime=src_atime, mtime=src_mtime, uid=src_uid, gid=src_gid, mode=src_mode)
                    src_url_str_to_check = _EncodeUrl(src_url_str[base_src_url_len:].replace('\\', '/'))
                    dst_url_str_would_copy_to = copy_helper.ConstructDstUrl(src_url=self.base_src_url, exp_src_url=StorageUrlFromString(src_url_str), src_url_names_container=True, have_multiple_srcs=True, has_multiple_top_level_srcs=False, exp_dst_url=self.base_dst_url, have_existing_dest_subdir=False, recursion_requested=self.recursion_requested).url_string
            if dst_url_str is None:
                if not self.sorted_dst_urls_it.IsEmpty():
                    dst_url_str, dst_size, _, dst_atime, dst_mtime, dst_mode, dst_uid, dst_gid, dst_crc32c, dst_md5 = self._ParseTmpFileLine(next(self.sorted_dst_urls_it))
                    dst_url_str_to_check = _EncodeUrl(dst_url_str[base_dst_url_len:].replace('\\', '/'))
            if out_of_src_items:
                break
            if dst_url_str is None or src_url_str_to_check < dst_url_str_to_check:
                yield RsyncDiffToApply(src_url_str, dst_url_str_would_copy_to, posix_attrs, DiffAction.COPY, src_size)
                src_url_str = None
            elif src_url_str_to_check > dst_url_str_to_check:
                if self.delete_extras:
                    yield RsyncDiffToApply(None, dst_url_str, POSIXAttributes(), DiffAction.REMOVE, None)
                dst_url_str = None
            else:
                if StorageUrlFromString(src_url_str).IsCloudUrl() and StorageUrlFromString(dst_url_str).IsFileUrl() and (src_mtime == NA_TIME):
                    src_mtime = src_time_created
                should_replace, has_src_mtime, has_dst_mtime = self._CompareObjects(src_url_str, src_size, src_mtime, src_crc32c, src_md5, dst_url_str, dst_size, dst_mtime, dst_crc32c, dst_md5)
                if should_replace:
                    yield RsyncDiffToApply(src_url_str, dst_url_str, posix_attrs, DiffAction.COPY, src_size)
                elif self.preserve_posix:
                    posix_attrs, needs_update = NeedsPOSIXAttributeUpdate(src_atime, dst_atime, src_mtime, dst_mtime, src_uid, dst_uid, src_gid, dst_gid, src_mode, dst_mode)
                    if needs_update:
                        yield RsyncDiffToApply(src_url_str, dst_url_str, posix_attrs, DiffAction.POSIX_SRC_TO_DST, src_size)
                elif has_src_mtime and (not has_dst_mtime):
                    yield RsyncDiffToApply(src_url_str, dst_url_str, posix_attrs, DiffAction.MTIME_SRC_TO_DST, src_size)
                src_url_str = None
                dst_url_str = None
        if not self.delete_extras:
            return
        if dst_url_str:
            yield RsyncDiffToApply(None, dst_url_str, POSIXAttributes(), DiffAction.REMOVE, None)
        for line in self.sorted_dst_urls_it:
            dst_url_str, _, _, _, _, _, _, _, _, _ = self._ParseTmpFileLine(line)
            yield RsyncDiffToApply(None, dst_url_str, POSIXAttributes(), DiffAction.REMOVE, None)