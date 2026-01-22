from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
from collections import namedtuple
import csv
import datetime
import errno
import gzip
import json
import logging
import mimetypes
from operator import attrgetter
import os
import pickle
import pyu2f
import random
import re
import shutil
import six
import stat
import subprocess
import tempfile
import textwrap
import time
import traceback
import six
from six.moves import xrange
from six.moves import range
from apitools.base.protorpclite import protojson
from boto import config
import crcmod
import gslib
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import CloudApi
from gslib.cloud_api import EncryptionException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.cloud_api import ResumableDownloadException
from gslib.cloud_api import ResumableUploadAbortException
from gslib.cloud_api import ResumableUploadException
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.cloud_api import ServiceException
from gslib.commands.compose import MAX_COMPOSE_ARITY
from gslib.commands.config import DEFAULT_PARALLEL_COMPOSITE_UPLOAD_COMPONENT_SIZE
from gslib.commands.config import DEFAULT_PARALLEL_COMPOSITE_UPLOAD_THRESHOLD
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_COMPONENT_SIZE
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_MAX_COMPONENTS
from gslib.commands.config import DEFAULT_SLICED_OBJECT_DOWNLOAD_THRESHOLD
from gslib.commands.config import DEFAULT_GZIP_COMPRESSION_LEVEL
from gslib.cs_api_map import ApiSelector
from gslib.daisy_chain_wrapper import DaisyChainWrapper
from gslib.exception import CommandException
from gslib.exception import HashMismatchException
from gslib.exception import InvalidUrlError
from gslib.file_part import FilePart
from gslib.parallel_tracker_file import GenerateComponentObjectPrefix
from gslib.parallel_tracker_file import ReadParallelUploadTrackerFile
from gslib.parallel_tracker_file import ValidateParallelCompositeTrackerData
from gslib.parallel_tracker_file import WriteComponentToParallelUploadTrackerFile
from gslib.parallel_tracker_file import WriteParallelUploadTrackerFile
from gslib.progress_callback import FileProgressCallbackHandler
from gslib.progress_callback import ProgressCallbackWithTimeout
from gslib.resumable_streaming_upload import ResumableStreamingJsonUploadWrapper
from gslib import storage_url
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import GenerationFromUrlAndString
from gslib.storage_url import IsCloudSubdirPlaceholder
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FileMessage
from gslib.thread_message import RetryableErrorMessage
from gslib.tracker_file import DeleteDownloadTrackerFiles
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import ENCRYPTION_UPLOAD_TRACKER_ENTRY
from gslib.tracker_file import GetDownloadStartByte
from gslib.tracker_file import GetTrackerFilePath
from gslib.tracker_file import GetUploadTrackerData
from gslib.tracker_file import RaiseUnwritableTrackerFileException
from gslib.tracker_file import ReadOrCreateDownloadTrackerFile
from gslib.tracker_file import SERIALIZATION_UPLOAD_TRACKER_ENTRY
from gslib.tracker_file import TrackerFileType
from gslib.tracker_file import WriteDownloadComponentTrackerFile
from gslib.tracker_file import WriteJsonDataToTrackerFile
from gslib.utils import parallelism_framework_util
from gslib.utils import stet_util
from gslib.utils import temporary_file_util
from gslib.utils import text_util
from gslib.utils.boto_util import GetJsonResumableChunkSize
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.boto_util import ResumableThreshold
from gslib.utils.boto_util import UsingCrcmodExtension
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.cloud_api_helper import GetDownloadSerializationData
from gslib.utils.constants import DEFAULT_FILE_BUFFER_SIZE
from gslib.utils.constants import MIN_SIZE_COMPUTE_LOGGING
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import CryptoKeyType
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.encryption_helper import FindMatchingCSEKInBotoConfig
from gslib.utils.encryption_helper import GetEncryptionKeyWrapper
from gslib.utils.hashing_helper import Base64EncodeHash
from gslib.utils.hashing_helper import CalculateB64EncodedMd5FromContents
from gslib.utils.hashing_helper import CalculateHashesFromContents
from gslib.utils.hashing_helper import CHECK_HASH_IF_FAST_ELSE_FAIL
from gslib.utils.hashing_helper import CHECK_HASH_NEVER
from gslib.utils.hashing_helper import ConcatCrc32c
from gslib.utils.hashing_helper import GetDownloadHashAlgs
from gslib.utils.hashing_helper import GetMd5
from gslib.utils.hashing_helper import GetUploadHashAlgs
from gslib.utils.hashing_helper import HashingFileUploadWrapper
from gslib.utils.metadata_util import ObjectIsGzipEncoded
from gslib.utils.parallelism_framework_util import AtomicDict
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
from gslib.utils.parallelism_framework_util import PutToQueueWithTimeout
from gslib.utils.posix_util import ATIME_ATTR
from gslib.utils.posix_util import ConvertDatetimeToPOSIX
from gslib.utils.posix_util import GID_ATTR
from gslib.utils.posix_util import MODE_ATTR
from gslib.utils.posix_util import MTIME_ATTR
from gslib.utils.posix_util import ParseAndSetPOSIXAttributes
from gslib.utils.posix_util import UID_ATTR
from gslib.utils.system_util import CheckFreeSpace
from gslib.utils.system_util import GetFileSize
from gslib.utils.system_util import GetStreamFromFileUrl
from gslib.utils.system_util import IS_WINDOWS
from gslib.utils.translation_helper import AddS3MarkerAclToObjectMetadata
from gslib.utils.translation_helper import CopyObjectMetadata
from gslib.utils.translation_helper import DEFAULT_CONTENT_TYPE
from gslib.utils.translation_helper import ObjectMetadataFromHeaders
from gslib.utils.translation_helper import PreconditionsFromHeaders
from gslib.utils.translation_helper import S3MarkerAclFromObjectMetadata
from gslib.utils.unit_util import DivideAndCeil
from gslib.utils.unit_util import HumanReadableToBytes
from gslib.utils.unit_util import MakeHumanReadable
from gslib.utils.unit_util import SECONDS_PER_DAY
from gslib.utils.unit_util import TEN_MIB
from gslib.wildcard_iterator import CreateWildcardIterator
def ConstructDstUrl(src_url, exp_src_url, src_url_names_container, have_multiple_srcs, has_multiple_top_level_srcs, exp_dst_url, have_existing_dest_subdir, recursion_requested, preserve_posix=False):
    """Constructs the destination URL for a given exp_src_url/exp_dst_url pair.

  Uses context-dependent naming rules that mimic Linux cp and mv behavior.

  Args:
    src_url: Source StorageUrl to be copied.
    exp_src_url: Single StorageUrl from wildcard expansion of src_url.
    src_url_names_container: True if src_url names a container (including the
        case of a wildcard-named bucket subdir (like gs://bucket/abc,
        where gs://bucket/abc/* matched some objects).
    have_multiple_srcs: True if this is a multi-source request. This can be
        true if src_url wildcard-expanded to multiple URLs or if there were
        multiple source URLs in the request.
    has_multiple_top_level_srcs: Same as have_multiple_srcs but measured
        before recursion.
    exp_dst_url: the expanded StorageUrl requested for the cp destination.
        Final written path is constructed from this plus a context-dependent
        variant of src_url.
    have_existing_dest_subdir: bool indicator whether dest is an existing
      subdirectory.
    recursion_requested: True if a recursive operation has been requested.
    preserve_posix: True if preservation of posix attributes has been requested.

  Returns:
    StorageUrl to use for copy.

  Raises:
    CommandException if destination object name not specified for
    source and source is a stream.
  """
    if exp_dst_url.IsFileUrl() and exp_dst_url.IsStream() and preserve_posix:
        raise CommandException('Cannot preserve POSIX attributes with a stream.')
    if _ShouldTreatDstUrlAsSingleton(src_url_names_container, have_multiple_srcs, have_existing_dest_subdir, exp_dst_url, recursion_requested):
        return exp_dst_url
    if exp_src_url.IsFileUrl() and (exp_src_url.IsStream() or exp_src_url.IsFifo()):
        if have_existing_dest_subdir:
            type_text = 'stream' if exp_src_url.IsStream() else 'named pipe'
            raise CommandException('Destination object name needed when source is a %s' % type_text)
        return exp_dst_url
    if not recursion_requested and (not have_multiple_srcs):
        src_final_comp = exp_src_url.object_name.rpartition(src_url.delim)[-1]
        return StorageUrlFromString('%s%s%s' % (exp_dst_url.url_string.rstrip(exp_dst_url.delim), exp_dst_url.delim, src_final_comp))
    if (have_multiple_srcs or src_url_names_container or (exp_dst_url.IsFileUrl() and exp_dst_url.IsDirectory())) and (not exp_dst_url.url_string.endswith(exp_dst_url.delim)):
        exp_dst_url = StorageUrlFromString('%s%s' % (exp_dst_url.url_string, exp_dst_url.delim))
    src_url_is_valid_parent = _IsUrlValidParentDir(src_url)
    if not src_url_is_valid_parent and has_multiple_top_level_srcs:
        raise InvalidUrlError('Presence of multiple top-level sources and invalid expanded URL make file name conflicts possible for URL: {}'.format(src_url))
    preserve_src_top_level_dirs = '**' not in src_url.versionless_url_string and src_url_is_valid_parent and (has_multiple_top_level_srcs or have_existing_dest_subdir)
    if preserve_src_top_level_dirs or (src_url_names_container and (exp_dst_url.IsCloudUrl() or exp_dst_url.IsDirectory())):
        src_url_path_sans_final_dir = GetPathBeforeFinalDir(src_url, exp_src_url)
        dst_key_name = exp_src_url.versionless_url_string[len(src_url_path_sans_final_dir):].lstrip(src_url.delim)
        if not preserve_src_top_level_dirs:
            dst_key_name = dst_key_name.partition(src_url.delim)[-1]
    else:
        dst_key_name = exp_src_url.object_name.rpartition(src_url.delim)[-1]
    if exp_dst_url.IsFileUrl() or _ShouldTreatDstUrlAsBucketSubDir(have_multiple_srcs, exp_dst_url, have_existing_dest_subdir, src_url_names_container, recursion_requested):
        if exp_dst_url.object_name and exp_dst_url.object_name.endswith(exp_dst_url.delim):
            dst_key_name = '%s%s%s' % (exp_dst_url.object_name.rstrip(exp_dst_url.delim), exp_dst_url.delim, dst_key_name)
        else:
            delim = exp_dst_url.delim if exp_dst_url.object_name else ''
            dst_key_name = '%s%s%s' % (exp_dst_url.object_name or '', delim, dst_key_name)
    new_exp_dst_url = exp_dst_url.Clone()
    new_exp_dst_url.object_name = dst_key_name.replace(src_url.delim, exp_dst_url.delim)
    return new_exp_dst_url