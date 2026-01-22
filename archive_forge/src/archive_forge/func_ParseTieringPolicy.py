from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.netapp import constants
from googlecloudsdk.api_lib.netapp import util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
def ParseTieringPolicy(self, volume, tiering_policy):
    """Parses Tiering Policy for Volume into a config.

    Args:
      volume: The Cloud NetApp Volume message object.
      tiering_policy: the tiering policy message object.

    Returns:
      Volume message populated with Tiering Policy values.
    """
    tiering_policy_message = self.messages.TieringPolicy()
    tiering_policy_message.tierAction = tiering_policy.get('tier-action')
    tiering_policy_message.coolingThresholdDays = tiering_policy.get('cooling-threshold-days')
    volume.tieringPolicy = tiering_policy_message