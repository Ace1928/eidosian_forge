from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QueuedResource(_messages.Message):
    """A QueuedResource represents a request for resources that will be placed
  in a queue and fulfilled when the necessary resources are available.

  Fields:
    createTime: Output only. The time when the QueuedResource was created.
    guaranteed: Optional. The Guaranteed tier
    name: Output only. Immutable. The name of the QueuedResource.
    queueingPolicy: Optional. The queueing policy of the QueuedRequest.
    reservationName: Optional. Name of the reservation in which the resource
      should be provisioned. Format:
      projects/{project}/locations/{zone}/reservations/{reservation}
    spot: Optional. The Spot tier.
    state: Output only. State of the QueuedResource request.
    tpu: Optional. Defines a TPU resource.
  """
    createTime = _messages.StringField(1)
    guaranteed = _messages.MessageField('Guaranteed', 2)
    name = _messages.StringField(3)
    queueingPolicy = _messages.MessageField('QueueingPolicy', 4)
    reservationName = _messages.StringField(5)
    spot = _messages.MessageField('Spot', 6)
    state = _messages.MessageField('QueuedResourceState', 7)
    tpu = _messages.MessageField('Tpu', 8)