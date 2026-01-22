from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TrinoJob(_messages.Message):
    """A Dataproc job for running Trino (https://trino.io/) queries. IMPORTANT:
  The Dataproc Trino Optional Component
  (https://cloud.google.com/dataproc/docs/concepts/components/trino) must be
  enabled when the cluster is created to submit a Trino job to the cluster.

  Messages:
    PropertiesValue: Optional. A mapping of property names to values. Used to
      set Trino session properties (https://trino.io/docs/current/sql/set-
      session.html) Equivalent to using the --session flag in the Trino CLI

  Fields:
    clientTags: Optional. Trino client tags to attach to this query
    continueOnFailure: Optional. Whether to continue executing queries if a
      query fails. The default value is false. Setting to true can be useful
      when executing independent parallel queries.
    loggingConfig: Optional. The runtime log config for job execution.
    outputFormat: Optional. The format in which query output will be
      displayed. See the Trino documentation for supported output formats
    properties: Optional. A mapping of property names to values. Used to set
      Trino session properties (https://trino.io/docs/current/sql/set-
      session.html) Equivalent to using the --session flag in the Trino CLI
    queryFileUri: The HCFS URI of the script that contains SQL queries.
    queryList: A list of queries.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Optional. A mapping of property names to values. Used to set Trino
    session properties (https://trino.io/docs/current/sql/set-session.html)
    Equivalent to using the --session flag in the Trino CLI

    Messages:
      AdditionalProperty: An additional property for a PropertiesValue object.

    Fields:
      additionalProperties: Additional properties of type PropertiesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PropertiesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    clientTags = _messages.StringField(1, repeated=True)
    continueOnFailure = _messages.BooleanField(2)
    loggingConfig = _messages.MessageField('LoggingConfig', 3)
    outputFormat = _messages.StringField(4)
    properties = _messages.MessageField('PropertiesValue', 5)
    queryFileUri = _messages.StringField(6)
    queryList = _messages.MessageField('QueryList', 7)