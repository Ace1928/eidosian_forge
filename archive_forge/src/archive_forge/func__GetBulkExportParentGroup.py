from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def _GetBulkExportParentGroup(parser, required=False, project_help='Project ID', org_help='Organization ID', folder_help='Folder ID'):
    """Creates parent flags for resource export.

  Args:
    parser:
    required:
    project_help:
    org_help:
    folder_help:

  Returns:
    Mutext group for resource export parent.
  """
    group = parser.add_group(mutex=True, required=required, help='`RESOURCE PARENT FLAG` - specify one of the following to determine the scope of exported resources.')
    group.add_argument('--organization', type=str, help=org_help)
    group.add_argument('--project', help=project_help)
    group.add_argument('--folder', type=str, help=folder_help)
    return group