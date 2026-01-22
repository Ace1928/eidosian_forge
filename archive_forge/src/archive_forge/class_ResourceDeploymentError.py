from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceDeploymentError(_messages.Message):
    """Message describing the error that occurred for the respective resource.

  Fields:
    errorMessage: Output only. Error details provided by deployment.
    httpCode: Output only. HTTP error code provided by the deployment.
  """
    errorMessage = _messages.StringField(1)
    httpCode = _messages.IntegerField(2, variant=_messages.Variant.INT32)