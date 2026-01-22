from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ExplainRequest(_messages.Message):
    """Request for explanations to be issued against a trained model.

  Fields:
    httpBody: Required. The explanation request body.
  """
    httpBody = _messages.MessageField('GoogleApiHttpBody', 1)