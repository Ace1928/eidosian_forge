from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsAbortFunctionUpgradeRequest(_messages.Message):
    """A CloudfunctionsProjectsLocationsFunctionsAbortFunctionUpgradeRequest
  object.

  Fields:
    abortFunctionUpgradeRequest: A AbortFunctionUpgradeRequest resource to be
      passed as the request body.
    name: Required. The name of the function for which upgrade should be
      aborted.
  """
    abortFunctionUpgradeRequest = _messages.MessageField('AbortFunctionUpgradeRequest', 1)
    name = _messages.StringField(2, required=True)