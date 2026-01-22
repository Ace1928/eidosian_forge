from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.database_migration import api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
def GetEngineName(self, profile):
    """Gets the SQL engine name from the connection profile.

    Args:
      profile: the connection profile

    Returns:
      A string representing the SQL engine
    """
    try:
        if profile.mysql:
            return 'MYSQL'
        if profile.cloudsql:
            return self._GetEngineFromCloudSql(profile.cloudsql)
        if profile.postgresql:
            return 'POSTGRES'
        if profile.alloydb:
            return ''
        if profile.oracle:
            return 'ORACLE'
        if profile.sqlserver:
            return 'SQLSERVER'
        return ''
    except AttributeError as _:
        return ''