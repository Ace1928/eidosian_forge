from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os
import re
import stat
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
import six
from six.moves import urllib
class HdfsUrl(BaseHdfsAndPosixUrl):
    """HDFS URL class providing parsing and convenience methods.

  Attributes:
    scheme (ProviderPrefix): This will always be "hdfs" for HdfsUrl.
    bucket_name (str): None for HdfsUrl.
    object_name (str): The file/directory path.
    generation (str): None for HdfsUrl.
  """

    def __init__(self, url_string):
        """Initialize HdfsUrl instance.

    Args:
      url_string (str): The string representing the filepath.
    """
        super(HdfsUrl, self).__init__(ProviderPrefix.HDFS, url_string)