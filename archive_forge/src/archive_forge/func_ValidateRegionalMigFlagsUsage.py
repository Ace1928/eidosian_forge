from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
from typing import Any
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.util.apis import arg_utils
def ValidateRegionalMigFlagsUsage(args, regional_flags_dests, igm_ref):
    """For zonal MIGs validate that user did not supply any RMIG-specific flags.

  Can be safely called from GA track for all flags, unknowns are ignored.

  Args:
    args: provided arguments.
    regional_flags_dests: list of RMIG-specific flag dests (names of the
      attributes used to store flag values in args).
    igm_ref: resource reference of the target IGM.
  """
    if igm_ref.Collection() == 'compute.regionInstanceGroupManagers':
        return
    for dest in regional_flags_dests:
        if args.IsKnownAndSpecified(dest):
            flag_name = args.GetFlag(dest)
            error_message = 'Flag %s may be specified for regional managed instance groups only.' % flag_name
            raise exceptions.InvalidArgumentException(parameter_name=flag_name, message=error_message)