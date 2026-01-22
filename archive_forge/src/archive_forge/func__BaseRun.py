from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from googlecloudsdk.api_lib.container import util as container_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as composer_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def _BaseRun(args):
    """Base operations for `get-config-connector-identity` run command."""
    container_util.CheckKubectlInstalled()
    cluster_id = 'krmapihost-' + args.name
    location = args.location
    project_id = args.project or properties.VALUES.core.project.GetOrFail()
    GetConfigConnectorIdentityForCluster(location, cluster_id, project_id)