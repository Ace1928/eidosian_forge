from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import os
import textwrap
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.spanner import database_operations
from googlecloudsdk.api_lib.spanner import databases
from googlecloudsdk.api_lib.spanner import instances
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.spanner import ddl_parser
from googlecloudsdk.command_lib.spanner import samples
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
def _create_db_op(instance_ref, database_id, statements, database_dialect):
    """Wrapper over databases.Create with error handling."""
    try:
        return databases.Create(instance_ref, database_id, statements, database_dialect=database_dialect)
    except apitools_exceptions.HttpConflictError:
        raise ValueError(textwrap.dedent("        Database '{database_id}' exists already. Delete it with:\n\n        $ gcloud spanner databases delete {database_id} --instance={instance_id}\n        ".format(database_id=database_id, instance_id=instance_ref.instancesId)))
    except apitools_exceptions.HttpError as ex:
        raise ValueError(json.loads(ex.content)['error']['message'])
    except Exception:
        raise ValueError("Failed to create database '{}'.".format(database_id))