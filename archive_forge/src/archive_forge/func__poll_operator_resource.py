from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.kuberun.core import events_constants
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.events import exceptions
from googlecloudsdk.command_lib.events import stages
from googlecloudsdk.command_lib.events import util
from googlecloudsdk.command_lib.kuberun.core.events import init_shared
from googlecloudsdk.core import log
from googlecloudsdk.core.console import progress_tracker
def _poll_operator_resource(client, operator_type, tracker):
    if operator_type == events_constants.Operator.KUBERUN:
        client.PollKubeRunResource(tracker)
    else:
        client.PollCloudRunResource(tracker)