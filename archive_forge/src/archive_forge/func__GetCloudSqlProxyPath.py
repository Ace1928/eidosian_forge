from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import errno
import os
import subprocess
import time
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.sql import api_util
from googlecloudsdk.api_lib.sql import constants
from googlecloudsdk.api_lib.sql import exceptions as sql_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import config
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files as file_utils
import six
def _GetCloudSqlProxyPath():
    """Determines the path to the cloud_sql_proxy binary."""
    sdk_bin_path = config.Paths().sdk_bin_path
    if sdk_bin_path:
        cloud_sql_proxy_path = os.path.join(sdk_bin_path, 'cloud_sql_proxy')
        if _IsCloudSqlProxyPresentInSdkBin(cloud_sql_proxy_path):
            return cloud_sql_proxy_path
    proxy_path = file_utils.FindExecutableOnPath('cloud_sql_proxy')
    if proxy_path:
        log.debug('Using cloud_sql_proxy found at [{path}]'.format(path=proxy_path))
        return proxy_path
    else:
        raise sql_exceptions.SqlProxyNotFound('A Cloud SQL Proxy SDK root could not be found, or access is denied.Please check your installation.')