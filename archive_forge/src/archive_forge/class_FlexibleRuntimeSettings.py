from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FlexibleRuntimeSettings(_messages.Message):
    """Runtime settings for the App Engine flexible environment.

  Fields:
    operatingSystem: The operating system of the application runtime.
    runtimeVersion: The runtime version of an App Engine flexible application.
  """
    operatingSystem = _messages.StringField(1)
    runtimeVersion = _messages.StringField(2)