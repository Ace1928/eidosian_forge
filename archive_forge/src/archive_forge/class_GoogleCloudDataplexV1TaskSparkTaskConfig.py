from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDataplexV1TaskSparkTaskConfig(_messages.Message):
    """User-specified config for running a Spark task.

  Fields:
    archiveUris: Optional. Cloud Storage URIs of archives to be extracted into
      the working directory of each executor. Supported file types: .jar,
      .tar, .tar.gz, .tgz, and .zip.
    fileUris: Optional. Cloud Storage URIs of files to be placed in the
      working directory of each executor.
    infrastructureSpec: Optional. Infrastructure specification for the
      execution.
    mainClass: The name of the driver's main class. The jar file that contains
      the class must be in the default CLASSPATH or specified in
      jar_file_uris. The execution args are passed in as a sequence of named
      process arguments (--key=value).
    mainJarFileUri: The Cloud Storage URI of the jar file that contains the
      main class. The execution args are passed in as a sequence of named
      process arguments (--key=value).
    pythonScriptFile: The Gcloud Storage URI of the main Python file to use as
      the driver. Must be a .py file. The execution args are passed in as a
      sequence of named process arguments (--key=value).
    sqlScript: The query text. The execution args are used to declare a set of
      script variables (set key="value";).
    sqlScriptFile: A reference to a query file. This can be the Cloud Storage
      URI of the query file or it can the path to a SqlScript Content. The
      execution args are used to declare a set of script variables (set
      key="value";).
  """
    archiveUris = _messages.StringField(1, repeated=True)
    fileUris = _messages.StringField(2, repeated=True)
    infrastructureSpec = _messages.MessageField('GoogleCloudDataplexV1TaskInfrastructureSpec', 3)
    mainClass = _messages.StringField(4)
    mainJarFileUri = _messages.StringField(5)
    pythonScriptFile = _messages.StringField(6)
    sqlScript = _messages.StringField(7)
    sqlScriptFile = _messages.StringField(8)