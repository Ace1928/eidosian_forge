from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppEngineVersionInfo(_messages.Message):
    """For display only. Metadata associated with an App Engine version.

  Fields:
    displayName: Name of an App Engine version.
    environment: App Engine execution environment for a version.
    runtime: Runtime of the App Engine version.
    uri: URI of an App Engine version.
  """
    displayName = _messages.StringField(1)
    environment = _messages.StringField(2)
    runtime = _messages.StringField(3)
    uri = _messages.StringField(4)