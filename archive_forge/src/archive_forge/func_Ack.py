from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import exceptions
def Ack(self, ack_ids, subscription_ref):
    """Acknowledges one or messages for a Subscription.

    Args:
      ack_ids (list[str]): List of ack ids for the messages being ack'd.
      subscription_ref (Resource): Relative name of the subscription for which
        to ack messages for.

    Returns:
      None:
    """
    ack_req = self.messages.PubsubProjectsSubscriptionsAcknowledgeRequest(acknowledgeRequest=self.messages.AcknowledgeRequest(ackIds=ack_ids), subscription=subscription_ref.RelativeName())
    return self._service.Acknowledge(ack_req)