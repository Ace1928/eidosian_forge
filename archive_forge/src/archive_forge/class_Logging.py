from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Logging(_messages.Message):
    """Logging configuration of the service.  The following example shows how
  to configure logs to be sent to the producer and consumer projects. In the
  example, the `library.googleapis.com/activity_history` log is sent to both
  the producer and consumer projects, whereas the
  `library.googleapis.com/purchase_history` log is only sent to the producer
  project:      monitored_resources:     - type: library.googleapis.com/branch
  labels:       - key: /city         description: The city where the library
  branch is located in.       - key: /name         description: The name of
  the branch.     logs:     - name: library.googleapis.com/activity_history
  labels:       - key: /customer_id     - name:
  library.googleapis.com/purchase_history     logging:
  producer_destinations:       - monitored_resource:
  library.googleapis.com/branch         logs:         -
  library.googleapis.com/activity_history         -
  library.googleapis.com/purchase_history       consumer_destinations:       -
  monitored_resource: library.googleapis.com/branch         logs:         -
  library.googleapis.com/activity_history

  Fields:
    consumerDestinations: Logging configurations for sending logs to the
      consumer project. There can be multiple consumer destinations, each one
      must have a different monitored resource type. A log can be used in at
      most one consumer destination.
    producerDestinations: Logging configurations for sending logs to the
      producer project. There can be multiple producer destinations, each one
      must have a different monitored resource type. A log can be used in at
      most one producer destination.
  """
    consumerDestinations = _messages.MessageField('LoggingDestination', 1, repeated=True)
    producerDestinations = _messages.MessageField('LoggingDestination', 2, repeated=True)