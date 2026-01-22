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
def _GetUpdateRequest(self, messages, commitment_ref, commitment_resource):
    return messages.ComputeRegionCommitmentsUpdateRequest(commitment=commitment_ref.Name(), commitmentResource=commitment_resource, paths=self._GetPaths(commitment_resource), project=commitment_ref.project, region=commitment_ref.region)