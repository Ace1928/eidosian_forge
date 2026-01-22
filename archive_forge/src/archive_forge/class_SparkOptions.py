from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkOptions(_messages.Message):
    """Options for a user-defined Spark routine.

  Messages:
    PropertiesValue: Configuration properties as a set of key/value pairs,
      which will be passed on to the Spark application. For more information,
      see [Apache Spark](https://spark.apache.org/docs/latest/index.html) and
      the [procedure option
      list](https://cloud.google.com/bigquery/docs/reference/standard-
      sql/data-definition-language#procedure_option_list).

  Fields:
    archiveUris: Archive files to be extracted into the working directory of
      each executor. For more information about Apache Spark, see [Apache
      Spark](https://spark.apache.org/docs/latest/index.html).
    connection: Fully qualified name of the user-provided Spark connection
      object. Format: ```"projects/{project_id}/locations/{location_id}/connec
      tions/{connection_id}"```
    containerImage: Custom container image for the runtime environment.
    fileUris: Files to be placed in the working directory of each executor.
      For more information about Apache Spark, see [Apache
      Spark](https://spark.apache.org/docs/latest/index.html).
    jarUris: JARs to include on the driver and executor CLASSPATH. For more
      information about Apache Spark, see [Apache
      Spark](https://spark.apache.org/docs/latest/index.html).
    mainClass: The fully qualified name of a class in jar_uris, for example,
      com.example.wordcount. Exactly one of main_class and main_jar_uri field
      should be set for Java/Scala language type.
    mainFileUri: The main file/jar URI of the Spark application. Exactly one
      of the definition_body field and the main_file_uri field must be set for
      Python. Exactly one of main_class and main_file_uri field should be set
      for Java/Scala language type.
    properties: Configuration properties as a set of key/value pairs, which
      will be passed on to the Spark application. For more information, see
      [Apache Spark](https://spark.apache.org/docs/latest/index.html) and the
      [procedure option
      list](https://cloud.google.com/bigquery/docs/reference/standard-
      sql/data-definition-language#procedure_option_list).
    pyFileUris: Python files to be placed on the PYTHONPATH for PySpark
      application. Supported file types: `.py`, `.egg`, and `.zip`. For more
      information about Apache Spark, see [Apache
      Spark](https://spark.apache.org/docs/latest/index.html).
    runtimeVersion: Runtime version. If not specified, the default runtime
      version is used.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Configuration properties as a set of key/value pairs, which will be
    passed on to the Spark application. For more information, see [Apache
    Spark](https://spark.apache.org/docs/latest/index.html) and the [procedure
    option list](https://cloud.google.com/bigquery/docs/reference/standard-
    sql/data-definition-language#procedure_option_list).

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
    connection = _messages.StringField(2)
    containerImage = _messages.StringField(3)
    fileUris = _messages.StringField(4, repeated=True)
    jarUris = _messages.StringField(5, repeated=True)
    mainClass = _messages.StringField(6)
    mainFileUri = _messages.StringField(7)
    properties = _messages.MessageField('PropertiesValue', 8)
    pyFileUris = _messages.StringField(9, repeated=True)
    runtimeVersion = _messages.StringField(10)