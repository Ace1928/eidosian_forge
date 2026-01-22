from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import re
import sys
import textwrap
from googlecloudsdk.calliope import walker
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import module_util
from googlecloudsdk.core.util import files
import six
def CliTreeDir():
    """The CLI tree default directory.

  This directory is part of the installation and its contents are managed
  by the installer/updater.

  Raises:
    SdkRootNotFoundError: If the SDK root directory does not exist.
    SdkDataCliNotFoundError: If the SDK root data CLI directory does not exist.

  Returns:
    The directory path.
  """
    paths = config.Paths()
    if paths.sdk_root is None:
        raise SdkRootNotFoundError('SDK root not found for this installation. CLI tree cannot be loaded or generated.')
    directory = os.path.join(paths.sdk_root, 'data', 'cli')
    if not os.path.isdir(directory):
        raise SdkDataCliNotFoundError('SDK root data CLI directory [{}] not found for this installation. CLI tree cannot be loaded or generated.'.format(directory))
    return directory