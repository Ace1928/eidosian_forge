from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubSubExportConfig(_messages.Message):
    """Configuration for a Pub/Sub export subscription.

  Enums:
    StateValueValuesEnum: Output only. An output-only field that indicates
      whether or not the subscription can receive messages.

  Fields:
    region: Optional. The GCP region to which messages will be published. If
      this is different from the region that messages were published, egress
      fees will be incurred. If the region is not specified, Pub/Sub will use
      the region to which the messages were originally published on a best-
      effort basis.
    serviceAccountEmail: Optional. The service account to use to publish to
      Pub/Sub. The subscription creator or updater that specifies this field
      must have `iam.serviceAccounts.actAs` permission on the service account.
      If not specified, the Pub/Sub [service
      agent](https://cloud.google.com/iam/docs/service-agents),
      service-{project_number}@gcp-sa-pubsub.iam.gserviceaccount.com, is used.
    state: Output only. An output-only field that indicates whether or not the
      subscription can receive messages.
    topic: Optional. The name of the topic to which to write data, of the form
      projects/{project_id}/topics/{topic_id}
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. An output-only field that indicates whether or not the
    subscription can receive messages.

    Values:
      STATE_UNSPECIFIED: Default value. This value is unused.
      ACTIVE: The subscription can actively send messages
      PERMISSION_DENIED: Cannot write to the destination because of permission
        denied errors.
      NOT_FOUND: Cannot write to the destination because it does not exist.
      SCHEMA_MISMATCH: Cannot write to the destination due to a schema
        mismatch.
      IN_TRANSIT_LOCATION_RESTRICTION: Cannot write to the destination because
        enforce_in_transit is set to true and the destination locations are
        not in the allowed regions.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        PERMISSION_DENIED = 2
        NOT_FOUND = 3
        SCHEMA_MISMATCH = 4
        IN_TRANSIT_LOCATION_RESTRICTION = 5
    region = _messages.StringField(1)
    serviceAccountEmail = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    topic = _messages.StringField(4)