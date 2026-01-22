from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PyFlinkJob(_messages.Message):
    """A Dataproc job for running Apache PyFlink (https://flink.apache.org/)
  applications on YARN.

  Messages:
    PropertiesValue: Optional. A mapping of property names to values, used to
      configure PyFlink. Properties that conflict with values set by the
      Dataproc API might be overwritten. Can include properties set in
      /etc/flink/conf/flink-defaults.conf and classes in user code.

  Fields:
    archiveUris: Optional. HCFS URIs of archives to be extracted into the
      working directory of each executor. Supported file types: .jar, .tar,
      .tar.gz, .tgz, and .zip.
    args: Optional. The arguments to pass to the driver. Do not include
      arguments, such as --conf, that can be set as job properties, since a
      collision might occur that causes an incorrect job submission.
    jarFileUris: Optional. HCFS URIs of jar files to add to the CLASSPATHs of
      the Python driver and tasks.
    loggingConfig: Optional. The runtime log config for job execution.
    mainPythonFileUri: Optional. The HCFS URI of the main Python file to use
      as the driver. Must be a .py file.
    properties: Optional. A mapping of property names to values, used to
      configure PyFlink. Properties that conflict with values set by the
      Dataproc API might be overwritten. Can include properties set in
      /etc/flink/conf/flink-defaults.conf and classes in user code.
    pythonFileUris: Optional. HCFS file URIs of Python files to pass to the
      PyFlink framework. Supported file types: .py, .egg, and .zip.
    pythonModule: Optional. The Python module that contains the PyFlink
      application entry point. This option must be used with python_file_uris
    pythonRequirements: Optional. The requirements.txt file which defines the
      third party dependencies of the PyFlink application
    savepointUri: Optional. HCFS URI of the savepoint which contains the last
      saved progress for this job.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PropertiesValue(_messages.Message):
        """Optional. A mapping of property names to values, used to configure
    PyFlink. Properties that conflict with values set by the Dataproc API
    might be overwritten. Can include properties set in /etc/flink/conf/flink-
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
    jarFileUris = _messages.StringField(3, repeated=True)
    loggingConfig = _messages.MessageField('LoggingConfig', 4)
    mainPythonFileUri = _messages.StringField(5)
    properties = _messages.MessageField('PropertiesValue', 6)
    pythonFileUris = _messages.StringField(7, repeated=True)
    pythonModule = _messages.StringField(8)
    pythonRequirements = _messages.StringField(9)
    savepointUri = _messages.StringField(10)