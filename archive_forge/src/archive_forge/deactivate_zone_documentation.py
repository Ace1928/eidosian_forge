from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dns import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dns import flags
from googlecloudsdk.core import properties
Deactivates a Cloud DNS peering zone.

  This command deactivates a Cloud DNS peering zone, removing any peering config
  and setting a deactivate time. Reponses sent to the deactivated zone will
  return REFUSED.

  ## EXAMPLES

  To deactivate a peering zone, run:

    $ {command} peering_zone_id
  