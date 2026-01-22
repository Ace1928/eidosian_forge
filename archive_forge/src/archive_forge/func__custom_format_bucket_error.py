from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.core import exceptions as core_exceptions
from six.moves import urllib
def _custom_format_bucket_error(self, match_bucket_resource_path, error_format):
    """Sets custom error formatting for buckets resource paths.

    Args:
      match_bucket_resource_path (re.Match): Match object that contains the
        result of searching regex BUCKET_RESOURCE_PATH_PATTERN in a resource
        path.
      error_format (str): Custom error format for buckets.
    """
    self.error_format = error_format
    self.payload.instance_name = match_bucket_resource_path.group('bucket')