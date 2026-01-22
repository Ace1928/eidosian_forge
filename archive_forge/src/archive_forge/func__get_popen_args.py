from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import textwrap
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.spanner import samples
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from surface.spanner.samples import init as samples_init
def _get_popen_args(project, appname, instance_id, database_id=None, port=None):
    """Get formatted args for server command."""
    if database_id is None:
        database_id = samples.get_db_id_for_app(appname)
    flags = ['--spanner_project_id={}'.format(project), '--spanner_instance_id={}'.format(instance_id), '--spanner_database_id={}'.format(database_id)]
    if port is not None:
        flags.append('--port={}'.format(port))
    if samples.get_database_dialect(appname) == databases.DATABASE_DIALECT_POSTGRESQL:
        flags.append('--spanner_use_pg')
    return flags