from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudHealthcareV1beta1DeidentifyOptions(_messages.Message):
    """Specifies additional options to apply to the base ProfileType.

  Fields:
    characterMaskConfig: Character mask config for CharacterMaskField.
    contextualDeid: Configure contextual de-id.
    cryptoHashConfig: Crypto hash config for CharacterMaskField.
    dateShiftConfig: Date shifting config for CharacterMaskField.
    keepExtensions: Configure keeping extensions by default.
  """
    characterMaskConfig = _messages.MessageField('CharacterMaskConfig', 1)
    contextualDeid = _messages.MessageField('ContextualDeidConfig', 2)
    cryptoHashConfig = _messages.MessageField('CryptoHashConfig', 3)
    dateShiftConfig = _messages.MessageField('DateShiftConfig', 4)
    keepExtensions = _messages.MessageField('KeepExtensionsConfig', 5)