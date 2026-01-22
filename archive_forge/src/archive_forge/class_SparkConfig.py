from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkConfig(_messages.Message):
    """Apache Spark (https://spark.apache.org) engine for an interactive
  session.

  Fields:
    archiveUris: Optional. HCFS URIs of archives to be extracted into the
      working directory of each executor. Supported file types: .jar, .tar,
      .tar.gz, .tgz, and .zip.
    fileUris: Optional. HCFS URIs of files to be placed in the working
      directory of each executor.
    jarFileUris: Optional. HCFS URIs of jar files to add to the classpath of
      the Spark driver and tasks.
  """
    archiveUris = _messages.StringField(1, repeated=True)
    fileUris = _messages.StringField(2, repeated=True)
    jarFileUris = _messages.StringField(3, repeated=True)