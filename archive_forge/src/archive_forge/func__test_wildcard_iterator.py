from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import os
import sys
import tempfile
import six
import boto
from boto.utils import get_utf8able_str
from gslib import project_id
from gslib import wildcard_iterator
from gslib.boto_translation import BotoTranslation
from gslib.cloud_api_delegator import CloudApiDelegator
from gslib.command_runner import CommandRunner
from gslib.cs_api_map import ApiMapConstants
from gslib.cs_api_map import ApiSelector
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.gcs_json_api import GcsJsonApi
from gslib.tests.mock_logging_handler import MockLoggingHandler
from gslib.tests.testcase import base
import gslib.tests.util as util
from gslib.tests.util import unittest
from gslib.tests.util import WorkingDirectory
from gslib.utils.constants import UTF8
from gslib.utils.text_util import print_to_fd
@classmethod
def _test_wildcard_iterator(cls, uri_or_str, exclude_tuple=None, debug=0):
    """Convenience method for instantiating a test instance of WildcardIterator.

    This makes it unnecessary to specify all the params of that class
    (like bucket_storage_uri_class=mock_storage_service.MockBucketStorageUri).
    Also, naming the factory method this way makes it clearer in the test code
    that WildcardIterator needs to be set up for testing.

    Args are same as for wildcard_iterator.wildcard_iterator(), except
    there are no class args for bucket_storage_uri_class or gsutil_api_class.

    Args:
      uri_or_str: StorageUri or string representing the wildcard string.
      exclude_tuple: (base_url, exclude_pattern), where base_url is
                     top-level URL to list; exclude_pattern is a regex
                     of paths to ignore during iteration.
      debug: debug level to pass to the underlying connection (0..3)

    Returns:
      WildcardIterator, over which caller can iterate.
    """
    uri_string = uri_or_str
    if hasattr(uri_or_str, 'uri'):
        uri_string = uri_or_str.uri
    return wildcard_iterator.CreateWildcardIterator(uri_string, cls.MakeGsUtilApi(debug), exclude_tuple=exclude_tuple)