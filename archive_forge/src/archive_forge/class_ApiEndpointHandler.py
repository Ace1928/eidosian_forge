from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApiEndpointHandler(_messages.Message):
    """Uses Google Cloud Endpoints to handle requests.

  Fields:
    scriptPath: Path to the script from the application root directory.
  """
    scriptPath = _messages.StringField(1)