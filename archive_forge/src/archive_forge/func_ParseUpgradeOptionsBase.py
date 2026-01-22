from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import container_command_util
from googlecloudsdk.command_lib.container import flags
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util.semver import SemVer
def ParseUpgradeOptionsBase(args):
    """Parses the flags provided with the cluster upgrade command."""
    return api_adapter.UpdateClusterOptions(version=args.cluster_version, update_master=args.master, update_nodes=not args.master, node_pool=args.node_pool, image_type=args.image_type, image=args.image, image_project=args.image_project)