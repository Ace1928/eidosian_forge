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
def Publish(self, request, global_params=None):
    """Adds one or more messages to the topic. Returns `NOT_FOUND` if the topic.
does not exist. The message payload must not be empty; it must contain
 either a non-empty data field, or at least one attribute.

      Args:
        request: (PubsubProjectsTopicsPublishRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (PublishResponse) The response message.
      """
    config = self.GetMethodConfig('Publish')
    return self._RunMethod(config, request, global_params=global_params)