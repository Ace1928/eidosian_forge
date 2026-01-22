from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2Connection(_messages.Message):
    """A data connection to allow DLP to profile data in locations that require
  additional configuration.

  Enums:
    StateValueValuesEnum: Required. The connection's state in its lifecycle.

  Fields:
    cloudSql: Connect to a Cloud SQL instance.
    errors: Output only. Set if status == ERROR, to provide additional
      details. Will store the last 10 errors sorted with the most recent
      first.
    name: Output only. Name of the connection:
      projects/{project}/locations/{location}/connections/{name}.
    state: Required. The connection's state in its lifecycle.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Required. The connection's state in its lifecycle.

    Values:
      CONNECTION_STATE_UNSPECIFIED: Unused
      MISSING_CREDENTIALS: DLP automatically created this connection during an
        initial scan, and it is awaiting full configuration by a user.
      AVAILABLE: A configured connection that has not encountered any errors.
      ERROR: A configured connection that encountered errors during its last
        use. It will not be used again until it is set to AVAILABLE. If the
        resolution requires external action, then a request to set the status
        to AVAILABLE will mark this connection for use. Otherwise, any changes
        to the connection properties will automatically mark it as AVAILABLE.
    """
        CONNECTION_STATE_UNSPECIFIED = 0
        MISSING_CREDENTIALS = 1
        AVAILABLE = 2
        ERROR = 3
    cloudSql = _messages.MessageField('GooglePrivacyDlpV2CloudSqlProperties', 1)
    errors = _messages.MessageField('GooglePrivacyDlpV2Error', 2, repeated=True)
    name = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)