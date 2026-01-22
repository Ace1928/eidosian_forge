from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
import time
from apitools.base.py import encoding
from boto import config
from gslib.cloud_api import EncryptionException
from gslib.command import Command
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.name_expansion import NameExpansionIterator
from gslib.name_expansion import SeekAheadNameExpansionIterator
from gslib.progress_callback import FileProgressCallbackHandler
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.thread_message import FileMessage
from gslib.utils.cloud_api_helper import GetCloudApiInstance
from gslib.utils.constants import NO_MAX
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import CryptoKeyType
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.encryption_helper import GetEncryptionKeyWrapper
from gslib.utils.encryption_helper import MAX_DECRYPTION_KEYS
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
from gslib.utils.system_util import StdinIterator
from gslib.utils.text_util import ConvertRecursiveToFlatWildcard
from gslib.utils.text_util import NormalizeStorageClass
from gslib.utils import text_util
from gslib.utils.translation_helper import PreconditionsFromHeaders
class _TransformTypes(object):
    """Enum class for valid transforms."""
    CRYPTO_KEY = 'crypto_key'
    STORAGE_CLASS = 'storage_class'