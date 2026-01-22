from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import org_policies
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_manager import org_policies_base
from googlecloudsdk.command_lib.resource_manager import org_policies_flags as flags
Turns off enforcement of boolean Organization Policy constraint.

  Turns off enforcement of a boolean Organization Policy constraint at
  the specified resource.

  ## EXAMPLES

  The following command disables enforcement of the Organization Policy boolean
  constraint `compute.disableSerialPortAccess` on project `foo-project`:

    $ {command} compute.disableSerialPortAccess --project=foo-project
  