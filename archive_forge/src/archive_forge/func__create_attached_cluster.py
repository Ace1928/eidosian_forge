from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.api_lib.container.gkemulticloud import attached as api_util
from googlecloudsdk.api_lib.container.gkemulticloud import locations as loc_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.container.attached import cluster_util
from googlecloudsdk.command_lib.container.attached import flags as attached_flags
from googlecloudsdk.command_lib.container.attached import resource_args
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.container.gkemulticloud import command_util
from googlecloudsdk.command_lib.container.gkemulticloud import constants
from googlecloudsdk.command_lib.container.gkemulticloud import endpoint_util
from googlecloudsdk.command_lib.container.gkemulticloud import errors
from googlecloudsdk.command_lib.container.gkemulticloud import flags
from googlecloudsdk.command_lib.run import pretty_print
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import retry
import six
def _create_attached_cluster(self, args, cluster_ref):
    cluster_client = api_util.ClustersClient()
    message = command_util.ClusterMessage(cluster_ref.attachedClustersId, action='Creating', kind=constants.ATTACHED)
    return command_util.Create(resource_ref=cluster_ref, resource_client=cluster_client, args=args, message=message, kind=constants.ATTACHED_CLUSTER_KIND)