from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ApiServerSourceSpec(_messages.Message):
    """The desired state of the ApiServerSource.

  Fields:
    ceOverrides: CloudEventOverrides defines overrides to control the output
      format and modifications of the event sent to the sink.
    mode: EventMode controls the format of the event. `Reference` sends a
      dataref event type for the resource under watch. `Resource` send the
      full resource lifecycle event. Defaults to `Reference`
    owner: ResourceOwner is an additional filter to only track resources that
      are owned by a specific resource type. If ResourceOwner matches
      Resources[n] then Resources[n] is allowed to pass the ResourceOwner
      filter.
    resources: Resource are the resources this source will track and send
      related lifecycle events from the Kubernetes ApiServer, with an optional
      label selector to help filter.
    serviceAccountName: ServiceAccountName is the k8s service account which
      binds to a google service account. This google service account has
      required permissions to poll from a Cloud Pub/Sub subscription. If not
      specified, defaults to use secret.
    sink: Sink is a reference to an object that will resolve to a uri to use
      as the sink.
  """
    ceOverrides = _messages.MessageField('CloudEventOverrides', 1)
    mode = _messages.StringField(2)
    owner = _messages.MessageField('APIVersionKind', 3)
    resources = _messages.MessageField('APIVersionKindSelector', 4, repeated=True)
    serviceAccountName = _messages.StringField(5)
    sink = _messages.MessageField('Destination', 6)