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
@staticmethod
def GetDatabaseInstances(limit=None, batch_size=None):
    """Gets SQL instances in a given project.

    Modifies current state of an individual instance to 'STOPPED' if
    activationPolicy is 'NEVER'.

    Args:
      limit: int, The maximum number of records to yield. None if all available
          records should be yielded.
      batch_size: int, The number of items to retrieve per request.

    Returns:
      List of yielded DatabaseInstancePresentation instances.
    """
    client = api_util.SqlClient(api_util.API_VERSION_DEFAULT)
    sql_client = client.sql_client
    sql_messages = client.sql_messages
    project_id = properties.VALUES.core.project.Get(required=True)
    params = {}
    if limit is not None:
        params['limit'] = limit
    default_batch_size = 1000
    params['batch_size'] = batch_size if batch_size is not None else default_batch_size
    yielded = list_pager.YieldFromList(sql_client.instances, sql_messages.SqlInstancesListRequest(project=project_id), **params)

    def YieldInstancesWithAModifiedState():
        for result in yielded:
            yield DatabaseInstancePresentation(result)
    return YieldInstancesWithAModifiedState()