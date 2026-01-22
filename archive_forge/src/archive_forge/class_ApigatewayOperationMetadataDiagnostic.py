from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayOperationMetadataDiagnostic(_messages.Message):
    """Diagnostic information from configuration processing.

  Fields:
    location: Location of the diagnostic.
    message: The diagnostic message.
  """
    location = _messages.StringField(1)
    message = _messages.StringField(2)