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
class RsyncCommand(Command):
    """Implementation of gsutil rsync command."""
    command_spec = Command.CreateCommandSpec('rsync', command_name_aliases=[], usage_synopsis=_SYNOPSIS, min_args=2, max_args=2, supported_sub_args='a:cCdenpPriRuUx:y:j:J', file_url_ok=True, provider_url_ok=False, urls_start_arg=0, gs_api_support=[ApiSelector.XML, ApiSelector.JSON], gs_default_api=ApiSelector.JSON, argparse_arguments=[CommandArgument.MakeNCloudOrFileURLsArgument(2)])
    help_spec = Command.HelpSpec(help_name='rsync', help_name_aliases=['sync', 'synchronize'], help_type='command_help', help_one_line_summary='Synchronize content of two buckets/directories', help_text=_DETAILED_HELP_TEXT, subcommand_help_text={})

    def get_gcloud_storage_args(self):
        ShimTranslatePredefinedAclSubOptForCopy(self.sub_opts)
        gcloud_command = ['storage', 'rsync']
        flag_keys = [flag for flag, _ in self.sub_opts]
        if '-e' not in flag_keys:
            gcloud_command += ['--no-ignore-symlinks']
            self.logger.warn('By default, gsutil copies file symlinks, but, by default, this command (run via the gcloud storage shim) does not copy any symlinks.')
        if '-P' in flag_keys:
            _, (source_path, destination_path) = self.ParseSubOpts(should_update_sub_opts_and_args=False)
            if StorageUrlFromString(source_path).IsCloudUrl() and StorageUrlFromString(destination_path).IsFileUrl():
                self.logger.warn('For preserving POSIX with rsync downloads, gsutil aborts if a single download will result in invalid destination POSIX. However, this command (run via the gcloud storage shim) will skip invalid copies and still perform valid copies.')
        gcloud_storage_map = GcloudStorageMap(gcloud_command=gcloud_command, flag_map={'-a': GcloudStorageFlag('--predefined-acl'), '-c': GcloudStorageFlag('--checksums-only'), '-C': GcloudStorageFlag('--continue-on-error'), '-d': GcloudStorageFlag('--delete-unmatched-destination-objects'), '-e': GcloudStorageFlag('--ignore-symlinks'), '-i': GcloudStorageFlag('--no-clobber'), '-J': GcloudStorageFlag('--gzip-in-flight-all'), '-j': GcloudStorageFlag('--gzip-in-flight'), '-n': GcloudStorageFlag('--dry-run'), '-P': GcloudStorageFlag('--preserve-posix'), '-p': GcloudStorageFlag('--preserve-acl'), '-R': GcloudStorageFlag('--recursive'), '-r': GcloudStorageFlag('--recursive'), '-U': GcloudStorageFlag('--skip-unsupported'), '-u': GcloudStorageFlag('--skip-if-dest-has-newer-mtime'), '-x': GcloudStorageFlag('--exclude')})
        return super().get_gcloud_storage_args(gcloud_storage_map)

    def _InsistContainer(self, url_str, treat_nonexistent_object_as_subdir):
        """Sanity checks that URL names an existing container.

    Args:
      url_str: URL string to check.
      treat_nonexistent_object_as_subdir: indicates if should treat a
                                          non-existent object as a subdir.

    Returns:
      URL for checked string.

    Raises:
      CommandException if url_str doesn't name an existing container.
    """
        url, have_existing_container = copy_helper.ExpandUrlToSingleBlr(url_str, self.gsutil_api, self.project_id, treat_nonexistent_object_as_subdir, logger=self.logger)
        if not have_existing_container:
            raise CommandException('arg (%s) does not name a directory, bucket, or bucket subdir.\nIf there is an object with the same path, please add a trailing\nslash to specify the directory.' % url_str)
        return url

    def RunCommand(self):
        """Command entry point for the rsync command."""
        self._ParseOpts()
        self.total_bytes_transferred = 0
        self.stats_lock = parallelism_framework_util.CreateLock()
        if not UsingCrcmodExtension():
            if self.compute_file_checksums:
                self.logger.warn(SLOW_CRCMOD_WARNING)
            else:
                self.logger.warn(SLOW_CRCMOD_RSYNC_WARNING)
        src_url = self._InsistContainer(self.args[0], False)
        dst_url = self._InsistContainer(self.args[1], True)
        is_daisy_chain = src_url.IsCloudUrl() and dst_url.IsCloudUrl() and (src_url.scheme != dst_url.scheme)
        LogPerformanceSummaryParams(has_file_src=src_url.IsFileUrl(), has_cloud_src=src_url.IsCloudUrl(), has_file_dst=dst_url.IsFileUrl(), has_cloud_dst=dst_url.IsCloudUrl(), is_daisy_chain=is_daisy_chain, uses_fan=self.parallel_operations, provider_types=[src_url.scheme, dst_url.scheme])
        self.source_metadata_fields = GetSourceFieldsNeededForCopy(dst_url.IsCloudUrl(), self.skip_unsupported_objects, self.preserve_acl, is_rsync=True, preserve_posix=self.preserve_posix_attrs)
        self.op_failure_count = 0
        shared_attrs = ('op_failure_count', 'total_bytes_transferred')
        for signal_num in GetCaughtSignals():
            RegisterSignalHandler(signal_num, _HandleSignals)
        process_count, thread_count = self._GetProcessAndThreadCount(process_count=None, thread_count=None, parallel_operations_override=self.ParallelOverrideReason.SPEED, print_macos_warning=False)
        copy_helper.TriggerReauthForDestinationProviderIfNecessary(dst_url, self.gsutil_api, worker_count=process_count * thread_count)
        diff_iterator = _DiffIterator(self, src_url, dst_url)
        seek_ahead_iterator = _SeekAheadDiffIterator(_AvoidChecksumAndListingDiffIterator(diff_iterator))
        self.logger.info('Starting synchronization...')
        start_time = time.time()
        try:
            self.Apply(_RsyncFunc, diff_iterator, _RsyncExceptionHandler, shared_attrs, arg_checker=_DiffToApplyArgChecker, fail_on_error=True, seek_ahead_iterator=seek_ahead_iterator)
        finally:
            CleanUpTempFiles()
        end_time = time.time()
        self.total_elapsed_time = end_time - start_time
        self.total_bytes_per_second = CalculateThroughput(self.total_bytes_transferred, self.total_elapsed_time)
        LogPerformanceSummaryParams(avg_throughput=self.total_bytes_per_second, total_elapsed_time=self.total_elapsed_time, total_bytes_transferred=self.total_bytes_transferred)
        if self.op_failure_count:
            plural_str = 's' if self.op_failure_count else ''
            raise CommandException('%d file%s/object%s could not be copied/removed.' % (self.op_failure_count, plural_str, plural_str))

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