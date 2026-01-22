from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectionProfile(_messages.Message):
    """A connection profile definition.

  Enums:
    ProviderValueValuesEnum: The database provider.
    StateValueValuesEnum: The current connection profile state (e.g. DRAFT,
      READY, or FAILED).

  Messages:
    LabelsValue: The resource labels for connection profile to use to annotate
      any related underlying resources such as Compute Engine VMs. An object
      containing a list of "key": "value" pairs. Example: `{ "name": "wrench",
      "mass": "1.3kg", "count": "3" }`.

  Fields:
    cloudsql: A CloudSQL database connection profile.
    createTime: Output only. The timestamp when the resource was created. A
      timestamp in RFC3339 UTC "Zulu" format, accurate to nanoseconds.
      Example: "2014-10-02T15:01:23.045123456Z".
    displayName: The connection profile display name.
    error: Output only. The error details in case of state FAILED.
    labels: The resource labels for connection profile to use to annotate any
      related underlying resources such as Compute Engine VMs. An object
      containing a list of "key": "value" pairs. Example: `{ "name": "wrench",
      "mass": "1.3kg", "count": "3" }`.
    mysql: A MySQL database connection profile.
    name: The name of this connection profile resource in the form of projects
      /{project}/locations/{location}/connectionProfiles/{connectionProfile}.
    provider: The database provider.
    state: The current connection profile state (e.g. DRAFT, READY, or
      FAILED).
    updateTime: Output only. The timestamp when the resource was last updated.
      A timestamp in RFC3339 UTC "Zulu" format, accurate to nanoseconds.
      Example: "2014-10-02T15:01:23.045123456Z".
  """

    class ProviderValueValuesEnum(_messages.Enum):
        """The database provider.

    Values:
      DATABASE_PROVIDER_UNSPECIFIED: The database provider is unknown.
      CLOUDSQL: CloudSQL runs the database.
      RDS: RDS runs the database.
    """
        DATABASE_PROVIDER_UNSPECIFIED = 0
        CLOUDSQL = 1
        RDS = 2

    class StateValueValuesEnum(_messages.Enum):
        """The current connection profile state (e.g. DRAFT, READY, or FAILED).

    Values:
      STATE_UNSPECIFIED: The state of the connection profile is unknown.
      DRAFT: The connection profile is in draft mode and fully editable.
      CREATING: The connection profile is being created.
      READY: The connection profile is ready.
      UPDATING: The connection profile is being updated.
      DELETING: The connection profile is being deleted.
      DELETED: The connection profile has been deleted.
      FAILED: The last action on the connection profile failed.
    """
        STATE_UNSPECIFIED = 0
        DRAFT = 1
        CREATING = 2
        READY = 3
        UPDATING = 4
        DELETING = 5
        DELETED = 6
        FAILED = 7

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """The resource labels for connection profile to use to annotate any
    related underlying resources such as Compute Engine VMs. An object
    containing a list of "key": "value" pairs. Example: `{ "name": "wrench",
    "mass": "1.3kg", "count": "3" }`.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    cloudsql = _messages.MessageField('CloudSqlConnectionProfile', 1)
    createTime = _messages.StringField(2)
    displayName = _messages.StringField(3)
    error = _messages.MessageField('Status', 4)
    labels = _messages.MessageField('LabelsValue', 5)
    mysql = _messages.MessageField('MySqlConnectionProfile', 6)
    name = _messages.StringField(7)
    provider = _messages.EnumField('ProviderValueValuesEnum', 8)
    state = _messages.EnumField('StateValueValuesEnum', 9)
    updateTime = _messages.StringField(10)