from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
import re
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.api_lib.compute.instance_groups.managed import autoscalers as autoscalers_api
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
def _PromptToAutoscaleGKENodeGroup(self, args):
    prompt_message = "You should not use Compute Engine's autoscaling feature on instance groups created by Kubernetes Engine."
    if re.match('^gke-.*-[0-9a-f]{1,8}-grp$', args.name):
        console_io.PromptContinue(message=prompt_message, default=False, cancel_on_no=True, cancel_string='Setting autoscaling aborted by user.')