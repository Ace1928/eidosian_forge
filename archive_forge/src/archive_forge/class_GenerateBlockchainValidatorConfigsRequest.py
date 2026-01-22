from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateBlockchainValidatorConfigsRequest(_messages.Message):
    """Generate a number of validator configurations from a common template.

  Fields:
    blockchainValidatorConfigTemplate: Required. The resources being created.
  """
    blockchainValidatorConfigTemplate = _messages.MessageField('BlockchainValidatorConfigTemplate', 1)