from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class CloudAuditLogsSourceSpec(_messages.Message):
    """The desired state of the CloudAuditLogsSource.

  Fields:
    ceOverrides: CloudEventOverrides defines overrides to control the output
      format and modifications of the event sent to the sink.
    methodName: The name of the service method or operation. For API calls,
      this should be the name of the API method. Required.
    project: Project is the ID of the Google Cloud Project that the
      CloudPubSubSource Topic exists in. If omitted, defaults to same as the
      cluster.
    resourceName: The resource or collection that is the target of the
      operation. The name is a scheme-less URI, not including the API service
      name.
    secret: Secret is the credential to use to create the Scheduler Job. If
      not specified, defaults to: Name: google-cloud-key Key: key.json
    serviceAccountName: ServiceAccountName is the k8s service account which
      binds to a google service account. This google service account has
      required permissions to poll from a Cloud Pub/Sub subscription. If not
      specified, defaults to use secret.
    serviceName: The GCP service providing audit logs. Required.
    sink: Sink is a reference to an object that will resolve to a uri to use
      as the sink.
  """
    ceOverrides = _messages.MessageField('CloudEventOverrides', 1)
    methodName = _messages.StringField(2)
    project = _messages.StringField(3)
    resourceName = _messages.StringField(4)
    secret = _messages.MessageField('SecretKeySelector', 5)
    serviceAccountName = _messages.StringField(6)
    serviceName = _messages.StringField(7)
    sink = _messages.MessageField('Destination', 8)