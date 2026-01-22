from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudfunctionsProjectsLocationsFunctionsSetupFunctionUpgradeConfigRequest(_messages.Message):
    """A
  CloudfunctionsProjectsLocationsFunctionsSetupFunctionUpgradeConfigRequest
  object.

  Fields:
    name: Required. The name of the function which should have configuration
      copied for upgrade.
    setupFunctionUpgradeConfigRequest: A SetupFunctionUpgradeConfigRequest
      resource to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    setupFunctionUpgradeConfigRequest = _messages.MessageField('SetupFunctionUpgradeConfigRequest', 2)