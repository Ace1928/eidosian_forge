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
def StartCloudSqlProxy(instance, port, seconds_to_timeout=10):
    """Starts the Cloud SQL Proxy for instance on the given port.

  Args:
    instance: The instance to start the proxy for.
    port: The port to bind the proxy to.
    seconds_to_timeout: Seconds to wait before timing out.

  Returns:
    The Process object corresponding to the Cloud SQL Proxy.

  Raises:
    CloudSqlProxyError: An error starting the Cloud SQL Proxy.
    SqlProxyNotFound: An error finding a Cloud SQL Proxy installation.
  """
    command_path = _GetCloudSqlProxyPath()
    args = ['-instances', '{}=tcp:{}'.format(instance.connectionName, port)]
    account = properties.VALUES.core.account.Get(required=True)
    args += ['-credential_file', config.Paths().LegacyCredentialsAdcPath(account)]
    proxy_args = execution_utils.ArgsForExecutableTool(command_path, *args)
    log.status.write('Starting Cloud SQL Proxy: [{args}]]\n'.format(args=' '.join(proxy_args)))
    try:
        proxy_process = subprocess.Popen(proxy_args, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except EnvironmentError as e:
        if e.errno == errno.ENOENT:
            raise sql_exceptions.CloudSqlProxyError('Failed to start Cloud SQL Proxy. Please make sure it is available in the PATH [{}]. Learn more about installing the Cloud SQL Proxy here: https://cloud.google.com/sql/docs/mysql/connect-admin-proxy#install. If you would like to report this issue, please run the following command: gcloud feedback'.format(command_path))
        raise
    return _WaitForProxyToStart(proxy_process, port, seconds_to_timeout)