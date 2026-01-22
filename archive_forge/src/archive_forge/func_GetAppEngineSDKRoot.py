from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import posixpath
import sys
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.util import platforms
from googlecloudsdk.third_party.appengine.api import client_deployinfo
import six
from six.moves import urllib
def GetAppEngineSDKRoot():
    """Gets the directory of the GAE SDK directory in the SDK.

  Raises:
    NoCloudSDKError: If there is no SDK root.
    NoAppengineSDKError: If the GAE SDK cannot be found.

  Returns:
    str, The path to the root of the GAE SDK within the Cloud SDK.
  """
    sdk_root = GetCloudSDKRoot()
    gae_sdk_dir = os.path.join(sdk_root, 'platform', 'google_appengine')
    if not os.path.isdir(gae_sdk_dir):
        raise NoAppengineSDKError()
    log.debug('Found App Engine SDK root: %s', gae_sdk_dir)
    return gae_sdk_dir