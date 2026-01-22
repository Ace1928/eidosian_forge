from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import os
import re
import string
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files as file_utils
from googlecloudsdk.core.util import platforms
import six
def _GetGsutilPath():
    """Determines the path to the gsutil binary."""
    sdk_bin_path = config.Paths().sdk_bin_path
    if not sdk_bin_path:
        gsutil_path = file_utils.FindExecutableOnPath('gsutil')
        if gsutil_path:
            log.debug('Using gsutil found at [{path}]'.format(path=gsutil_path))
            return gsutil_path
        else:
            raise GsutilError('A path to the storage client `gsutil` could not be found. Please check your SDK installation.')
    return os.path.join(sdk_bin_path, 'gsutil')