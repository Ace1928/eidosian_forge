from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.bigtable import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.bigtable import arguments
from googlecloudsdk.core import resources
def _TransformCluster(resource):
    """Get Cluster ID from backup name."""
    backup_name = resource.get('name')
    results = backup_name.split('/')
    cluster_name = results[-3]
    return cluster_name