from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HadoopJob(_messages.Message):
    """A Dataproc job for running Apache Hadoop MapReduce
  (https://hadoop.apache.org/docs/current/hadoop-mapreduce-client/hadoop-
  mapreduce-client-core/MapReduceTutorial.html) jobs on Apache Hadoop YARN
  (https://hadoop.apache.org/docs/r2.7.1/hadoop-yarn/hadoop-yarn-
  site/YARN.html).

  Messages:
    PropertiesValue: Optional. A mapping of property names to values, used to
      configure Hadoop. Properties that conflict with values set by the
      Dataproc API might be overwritten. Can include properties set in
      /etc/hadoop/conf/*-site and classes in user code.

  Fields:
    archiveUris: Optional. HCFS URIs of archives to be extracted in the
      working directory of Hadoop drivers and tasks. Supported file types:
      .jar, .tar, .tar.gz, .tgz, or .zip.
    args: Optional. The arguments to pass to the driver. Do not include
      arguments, such as -libjars or -Dfoo=bar, that can be set as job
      properties, since a collision might occur that causes an incorrect job
      submission.
    fileUris: Optional. HCFS (Hadoop Compatible Filesystem) URIs of files to
      be copied to the working directory of Hadoop drivers and distributed
      tasks. Useful for naively parallel tasks.
    jarFileUris: Optional. Jar file URIs to add to the CLASSPATHs of the
      Hadoop driver and tasks.
    loggingConfig: Optional. The runtime log config for job execution.
    mainClass: The name of the driver's main class. The jar file containing
      the class must be in the default CLASSPATH or specified in
      jar_file_uris.
    mainJarFileUri: The HCFS URI of the jar file containing the main class.
      Examples: 'gs://foo-bucket/analytics-binaries/extract-useful-metrics-
      mr.jar' 'hdfs:/tmp/test-samples/custom-wordcount.jar'
      'file:///home/usr/lib/hadoop-mapreduce/hadoop-mapreduce-examples.jar'
    properties: Optional. A mapping of property names to values, used to
      configure Hadoop. Properties that conflict with values set by the
      Dataproc API might be overwritten. Can include properties set in
      /etc/hadoop/conf/*-site and classes in user code.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Optional. A mapping of property names to values, used to configure
    Hadoop. Properties that conflict with values set by the Dataproc API might
    be overwritten. Can include properties set in /etc/hadoop/conf/*-site and
    classes in user code.

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