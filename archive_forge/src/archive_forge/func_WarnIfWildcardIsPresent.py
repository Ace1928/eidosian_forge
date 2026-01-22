from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os.path
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import transfer
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.command_lib.util import gcloudignore
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
def WarnIfWildcardIsPresent(path, flag_name):
    """Logs deprecation warning if gsutil wildcards are in args."""
    if path and ('*' in path or '?' in path or re.search('\\[.*\\]', path)):
        log.warning('Use of gsutil wildcards is no longer supported in {0}. Set the storage/use_gsutil property to get the old behavior back temporarily. However, this property will eventually be removed.'.format(flag_name))