from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PigJob(_messages.Message):
    """A Dataproc job for running Apache Pig (https://pig.apache.org/) queries
  on YARN.

  Messages:
    PropertiesValue: Optional. A mapping of property names to values, used to
      configure Pig. Properties that conflict with values set by the Dataproc
      API might be overwritten. Can include properties set in
      /etc/hadoop/conf/*-site.xml, /etc/pig/conf/pig.properties, and classes
      in user code.
    ScriptVariablesValue: Optional. Mapping of query variable names to values
      (equivalent to the Pig command: name=[value]).

  Fields:
    continueOnFailure: Optional. Whether to continue executing queries if a
      query fails. The default value is false. Setting to true can be useful
      when executing independent parallel queries.
    jarFileUris: Optional. HCFS URIs of jar files to add to the CLASSPATH of
      the Pig Client and Hadoop MapReduce (MR) tasks. Can contain Pig UDFs.
    loggingConfig: Optional. The runtime log config for job execution.
    properties: Optional. A mapping of property names to values, used to
      configure Pig. Properties that conflict with values set by the Dataproc
      API might be overwritten. Can include properties set in
      /etc/hadoop/conf/*-site.xml, /etc/pig/conf/pig.properties, and classes
      in user code.
    queryFileUri: The HCFS URI of the script that contains the Pig queries.
    queryList: A list of queries.
    scriptVariables: Optional. Mapping of query variable names to values
      (equivalent to the Pig command: name=[value]).
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Optional. A mapping of property names to values, used to configure
    Pig. Properties that conflict with values set by the Dataproc API might be
    overwritten. Can include properties set in /etc/hadoop/conf/*-site.xml,
    /etc/pig/conf/pig.properties, and classes in user code.

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

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ScriptVariablesValue(_messages.Message):
        """Optional. Mapping of query variable names to values (equivalent to the
    Pig command: name=[value]).

    Messages:
      AdditionalProperty: An additional property for a ScriptVariablesValue
        object.

    Fields:
      additionalProperties: Additional properties of type ScriptVariablesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ScriptVariablesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    continueOnFailure = _messages.BooleanField(1)
    jarFileUris = _messages.StringField(2, repeated=True)
    loggingConfig = _messages.MessageField('LoggingConfig', 3)
    properties = _messages.MessageField('PropertiesValue', 4)
    queryFileUri = _messages.StringField(5)
    queryList = _messages.MessageField('QueryList', 6)
    scriptVariables = _messages.MessageField('ScriptVariablesValue', 7)