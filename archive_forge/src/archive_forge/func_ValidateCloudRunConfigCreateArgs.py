from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def ValidateCloudRunConfigCreateArgs(cloud_run_config_args, addons_args):
    """Validates flags specifying Cloud Run config for create.

  Args:
    cloud_run_config_args: parsed commandline arguments for --cloud-run-config.
    addons_args: parsed commandline arguments for --addons.

  Raises:
    InvalidArgumentException: when load-balancer-type is not EXTERNAL nor
    INTERNAL,
    or --addons=CloudRun is not specified
  """
    if cloud_run_config_args:
        load_balancer_type = cloud_run_config_args.get('load-balancer-type', '')
        if load_balancer_type not in ['EXTERNAL', 'INTERNAL']:
            raise exceptions.InvalidArgumentException('--kuberun-config', 'load-balancer-type is either EXTERNAL or INTERNALe.g. --kuberun-config load-balancer-type=EXTERNAL')
        if all((v not in addons_args for v in api_adapter.CLOUDRUN_ADDONS)):
            raise exceptions.InvalidArgumentException('--kuberun-config', '--addon=KubeRun must be specified when --kuberun-config is given')