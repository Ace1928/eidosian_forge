from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LoggingProjectsLocationsOperationsApproveRedactionRequest(_messages.Message):
    """A LoggingProjectsLocationsOperationsApproveRedactionRequest object.

  Fields:
    name: Required. Name of the redaction operation.For example:"projects/my-
      project/locations/global/operations/my-operation"
  """
    name = _messages.StringField(1, required=True)