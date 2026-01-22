from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudPubSubSourceSpec(_messages.Message):
    """The desired state of the CloudPubSubSource.

  Fields:
    ackDeadline: AckDeadline is the default maximum time after a subscriber
      receives a message before the subscriber should acknowledge the message.
      Defaults to 30 seconds ('30s'). +optional
    ceOverrides: CloudEventOverrides defines overrides to control the output
      format and modifications of the event sent to the sink.
    project: Project is the ID of the Google Cloud Project that the
      CloudPubSubSource Topic exists in. If omitted, defaults to same as the
      cluster.
    retainAckedMessages: RetainAckedMessages defines whether to retain
      acknowledged messages. If true, acknowledged messages will not be
      expunged until they fall out of the RetentionDuration window.
    retentionDuration: RetentionDuration defines how long to retain messages
      in backlog, from the time of publish. If RetainAckedMessages is true,
      this duration affects the retention of acknowledged messages, otherwise
      only unacknowledged messages are retained. Cannot be longer than 7 days
      or shorter than 10 minutes. Defaults to 7 days ('7d'). +optional
    secret: Secret is the credential to use to create the Scheduler Job. If
      not specified, defaults to: Name: google-cloud-key Key: key.json
    serviceAccountName: ServiceAccountName is the k8s service account which
      binds to a google service account. This google service account has
      required permissions to poll from a Cloud Pub/Sub subscription. If not
      specified, defaults to use secret.
    sink: Sink is a reference to an object that will resolve to a domain name
      or a URI directly to use as the sink.
    topic: Topic is the ID of the CloudPubSubSource Topic to Subscribe to. It
      must be in the form of the unique identifier within the project, not the
      entire name. E.g. it must be 'laconia', not 'projects/my-
      proj/topics/laconia'.
  """
    ackDeadline = _messages.StringField(1)
    ceOverrides = _messages.MessageField('CloudEventOverrides', 2)
    project = _messages.StringField(3)
    retainAckedMessages = _messages.BooleanField(4)
    retentionDuration = _messages.StringField(5)
    secret = _messages.MessageField('SecretKeySelector', 6)
    serviceAccountName = _messages.StringField(7)
    sink = _messages.MessageField('Destination', 8)
    topic = _messages.StringField(9)