from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkJob(_messages.Message):
    """A Dataproc job for running Apache Spark (https://spark.apache.org/)
  applications on YARN.

  Messages:
    PropertiesValue: Optional. A mapping of property names to values, used to
      configure Spark. Properties that conflict with values set by the
      Dataproc API might be overwritten. Can include properties set in
      /etc/spark/conf/spark-defaults.conf and classes in user code.

  Fields:
    archiveUris: Optional. HCFS URIs of archives to be extracted into the
      working directory of each executor. Supported file types: .jar, .tar,
      .tar.gz, .tgz, and .zip.
    args: Optional. The arguments to pass to the driver. Do not include
      arguments, such as --conf, that can be set as job properties, since a
      collision may occur that causes an incorrect job submission.
    fileUris: Optional. HCFS URIs of files to be placed in the working
      directory of each executor. Useful for naively parallel tasks.
    jarFileUris: Optional. HCFS URIs of jar files to add to the CLASSPATHs of
      the Spark driver and tasks.
    loggingConfig: Optional. The runtime log config for job execution.
    mainClass: The name of the driver's main class. The jar file that contains
      the class must be in the default CLASSPATH or specified in
      SparkJob.jar_file_uris.
    mainJarFileUri: The HCFS URI of the jar file that contains the main class.
    properties: Optional. A mapping of property names to values, used to
      configure Spark. Properties that conflict with values set by the
      Dataproc API might be overwritten. Can include properties set in
      /etc/spark/conf/spark-defaults.conf and classes in user code.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Optional. A mapping of property names to values, used to configure
    Spark. Properties that conflict with values set by the Dataproc API might
    be overwritten. Can include properties set in /etc/spark/conf/spark-
    defaults.conf and classes in user code.

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
    archiveUris = _messages.StringField(1, repeated=True)
    args = _messages.StringField(2, repeated=True)
    fileUris = _messages.StringField(3, repeated=True)
    jarFileUris = _messages.StringField(4, repeated=True)
    loggingConfig = _messages.MessageField('LoggingConfig', 5)
    mainClass = _messages.StringField(6)
    mainJarFileUri = _messages.StringField(7)
    properties = _messages.MessageField('PropertiesValue', 8)