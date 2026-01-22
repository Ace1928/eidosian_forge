from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.commitments import flags
def _GetPaths(self, commitment_resource):
    paths = []
    if commitment_resource.autoRenew is not None:
        paths.append('autoRenew')
    if commitment_resource.plan is not None:
        paths.append('plan')
    return paths