from __future__ import absolute_import
import os
import platform
import sys
from apitools.base.py import base_api
import gslib.third_party.pubsub_apitools.pubsub_v1_messages as messages
import gslib
from gslib.metrics import MetricsCollector
from gslib.utils import system_util
from the subscription.
def ModifyAckDeadline(self, request, global_params=None):
    """Modifies the ack deadline for a specific message. This method is useful.
to indicate that more time is needed to process a message by the
subscriber, or to make the message available for redelivery if the
processing was interrupted. Note that this does not modify the
subscription-level `ackDeadlineSeconds` used for subsequent messages.

      Args:
        request: (PubsubProjectsSubscriptionsModifyAckDeadlineRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Empty) The response message.
      """
    config = self.GetMethodConfig('ModifyAckDeadline')
    return self._RunMethod(config, request, global_params=global_params)