from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.security_policies import flags
Remove a layer7 ddos defense threshold config from a Compute Engine security policy.

  *{command}* is used to remove layer7 ddos defense threshold configs from security policies.

  ## EXAMPLES

  To remove a layer7 ddos defense threshold config run the following command:

    $ {command} NAME \
       --threshold-config-name=my-threshold-config-name
  