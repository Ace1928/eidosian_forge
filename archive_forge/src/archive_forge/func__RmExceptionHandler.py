from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import errno
import itertools
import logging
import os
import time
import traceback
from apitools.base.py import encoding
from gslib import gcs_json_api
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.metrics import LogPerformanceSummaryParams
from gslib.name_expansion import CopyObjectsIterator
from gslib.name_expansion import DestinationInfo
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import NameExpansionIteratorDestinationTuple
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.storage_url import ContainsWildcard
from gslib.storage_url import IsCloudSubdirPlaceholder
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import cat_helper
from gslib.utils import copy_helper
from gslib.utils import parallelism_framework_util
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.constants import DEBUGLEVEL_DUMP_REQUESTS
from gslib.utils.constants import NO_MAX
from gslib.utils.copy_helper import CreateCopyHelperOpts
from gslib.utils.copy_helper import GetSourceFieldsNeededForCopy
from gslib.utils.copy_helper import GZIP_ALL_FILES
from gslib.utils.copy_helper import ItemExistsError
from gslib.utils.copy_helper import Manifest
from gslib.utils.copy_helper import SkipUnsupportedObjectError
from gslib.utils.posix_util import ConvertModeToBase8
from gslib.utils.posix_util import DeserializeFileAttributesFromObjectMetadata
from gslib.utils.posix_util import InitializePreservePosixData
from gslib.utils.posix_util import POSIXAttributes
from gslib.utils.posix_util import SerializeFileAttributesToObjectMetadata
from gslib.utils.posix_util import ValidateFilePermissionAccess
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.system_util import GetStreamFromFileUrl
from gslib.utils.system_util import StdinIterator
from gslib.utils.system_util import StdinIteratorCls
from gslib.utils.text_util import NormalizeStorageClass
from gslib.utils.text_util import RemoveCRLFFromString
from gslib.utils.unit_util import CalculateThroughput
from gslib.utils.unit_util import MakeHumanReadable
def _RmExceptionHandler(cls, e):
    """Simple exception handler to allow post-completion status."""
    cls.logger.error(str(e))