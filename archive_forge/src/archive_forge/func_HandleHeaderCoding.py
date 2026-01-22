from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import difflib
import logging
import os
import pkgutil
import sys
import textwrap
import time
import six
from six.moves import input
import boto
from boto import config
from boto.storage_uri import BucketStorageUri
import gslib
from gslib import metrics
from gslib.cloud_api_delegator import CloudApiDelegator
from gslib.command import Command
from gslib.command import CreateOrGetGsutilLogger
from gslib.command import GetFailureCount
from gslib.command import OLD_ALIAS_MAP
from gslib.command import ShutDownGsutil
import gslib.commands
from gslib.cs_api_map import ApiSelector
from gslib.cs_api_map import GsutilApiClassMapFactory
from gslib.cs_api_map import GsutilApiMapFactory
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.exception import CommandException
from gslib.gcs_json_api import GcsJsonApi
from gslib.no_op_credentials import NoOpCredentials
from gslib.tab_complete import MakeCompleter
from gslib.utils import boto_util
from gslib.utils import shim_util
from gslib.utils import system_util
from gslib.utils.constants import RELEASE_NOTES_URL
from gslib.utils.constants import UTF8
from gslib.utils.metadata_util import IsCustomMetadataHeader
from gslib.utils.parallelism_framework_util import CheckMultiprocessingAvailableAndInit
from gslib.utils.text_util import CompareVersions
from gslib.utils.text_util import InsistAsciiHeader
from gslib.utils.text_util import InsistAsciiHeaderValue
from gslib.utils.text_util import print_to_fd
from gslib.utils.unit_util import SECONDS_PER_DAY
from gslib.utils.update_util import LookUpGsutilVersion
from gslib.utils.update_util import GsutilPubTarball
def HandleHeaderCoding(headers):
    """Handles coding of headers and their values. Alters the dict in-place.

  Converts a dict of headers and their values to their appropriate types. We
  ensure that all headers and their values will contain only ASCII characters,
  with the exception of custom metadata header values; these values may contain
  Unicode characters, and thus if they are not already unicode-type objects,
  we attempt to decode them to Unicode using UTF-8 encoding.

  Args:
    headers: A dict mapping headers to their values. All keys and values must
        be either str or unicode objects.

  Raises:
    CommandException: If a header or its value cannot be encoded in the
        required encoding.
  """
    if not headers:
        return
    for key in headers:
        InsistAsciiHeader(key)
        if IsCustomMetadataHeader(key):
            if not isinstance(headers[key], six.text_type):
                try:
                    headers[key] = headers[key].decode(UTF8)
                except UnicodeDecodeError:
                    raise CommandException('\n'.join(textwrap.wrap('Invalid encoding for header value (%s: %s). Values must be decodable as Unicode. NOTE: the value printed above replaces the problematic characters with a hex-encoded printable representation. For more details (including how to convert to a gsutil-compatible encoding) see `gsutil help encoding`.' % (repr(key), repr(headers[key])))))
        else:
            InsistAsciiHeaderValue(key, headers[key])