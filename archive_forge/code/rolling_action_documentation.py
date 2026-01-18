from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import managed_instance_groups_utils
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.instance_groups import flags as instance_groups_flags
from googlecloudsdk.command_lib.compute.managed_instance_groups import update_instances_utils
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.util import times
import six
Validates and fixes update policy for patching stateful IGM.

  Updating stateful IGMs requires maxSurge=0 and replacementMethod=RECREATE.
  If the field has the value set, it is validated.
  If the field has the value not set, it is being set.

  Args:
    update_policy: Update policy to be validated
    igm_info: Full resource of IGM being validated
    client: The compute API client
    args: argparse namespace used to select used version
  