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
def AddPodAutoscalingDirectMetricsOptInFlag(parser):
    """Adds a --pod-autoscaling-direct-metrics-opt-in flag to the given parser."""
    parser.add_argument('--pod-autoscaling-direct-metrics-opt-in', default=None, action='store_true', hidden=True, help='When specified, the cluster will use the pod autoscaling direct metrics collection feature. Otherwise the cluster will use the feature or will not, depending on the cluster version.')