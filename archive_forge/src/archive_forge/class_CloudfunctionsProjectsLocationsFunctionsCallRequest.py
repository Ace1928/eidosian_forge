from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsCallRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsFunctionsCallRequest object.

  Fields:
    callFunctionRequest: A CallFunctionRequest resource to be passed as the
      request body.
    name: Required. The name of the function to be called.
  """
    callFunctionRequest = _messages.MessageField('CallFunctionRequest', 1)
    name = _messages.StringField(2, required=True)