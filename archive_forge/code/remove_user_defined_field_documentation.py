from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.security_policies import flags
Remove a user defined field from a Compute Engine security policy.

  *{command}* is used to remove user defined fields from security policies.

  ## EXAMPLES

  To remove a user defined field run this:

    $ {command} SECURITY_POLICY --user-defined-field-name=my-field
  