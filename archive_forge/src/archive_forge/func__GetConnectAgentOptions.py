from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container.fleet import gkehub_api_adapter
from googlecloudsdk.api_lib.container.fleet import gkehub_api_util
from googlecloudsdk.command_lib.container.fleet import api_util
from googlecloudsdk.command_lib.container.fleet import kube_util
from googlecloudsdk.command_lib.projects import util as p_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
def _GetConnectAgentOptions(args, upgrade, namespace, image_pull_secret_data, membership_ref):
    return gkehub_api_adapter.ConnectAgentOption(name=args.MEMBERSHIP_NAME, proxy=args.proxy or '', namespace=namespace, is_upgrade=upgrade, version=args.version or '', registry=args.docker_registry or '', image_pull_secret_content=image_pull_secret_data or '', membership_ref=membership_ref)