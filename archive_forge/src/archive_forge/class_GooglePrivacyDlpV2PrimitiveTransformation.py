from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2PrimitiveTransformation(_messages.Message):
    """A rule for transforming a value.

  Fields:
    bucketingConfig: Bucketing
    characterMaskConfig: Mask
    cryptoDeterministicConfig: Deterministic Crypto
    cryptoHashConfig: Crypto
    cryptoReplaceFfxFpeConfig: Ffx-Fpe
    dateShiftConfig: Date Shift
    fixedSizeBucketingConfig: Fixed size bucketing
    redactConfig: Redact
    replaceConfig: Replace with a specified value.
    replaceDictionaryConfig: Replace with a value randomly drawn (with
      replacement) from a dictionary.
    replaceWithInfoTypeConfig: Replace with infotype
    timePartConfig: Time extraction
  """
    bucketingConfig = _messages.MessageField('GooglePrivacyDlpV2BucketingConfig', 1)
    characterMaskConfig = _messages.MessageField('GooglePrivacyDlpV2CharacterMaskConfig', 2)
    cryptoDeterministicConfig = _messages.MessageField('GooglePrivacyDlpV2CryptoDeterministicConfig', 3)
    cryptoHashConfig = _messages.MessageField('GooglePrivacyDlpV2CryptoHashConfig', 4)
    cryptoReplaceFfxFpeConfig = _messages.MessageField('GooglePrivacyDlpV2CryptoReplaceFfxFpeConfig', 5)
    dateShiftConfig = _messages.MessageField('GooglePrivacyDlpV2DateShiftConfig', 6)
    fixedSizeBucketingConfig = _messages.MessageField('GooglePrivacyDlpV2FixedSizeBucketingConfig', 7)
    redactConfig = _messages.MessageField('GooglePrivacyDlpV2RedactConfig', 8)
    replaceConfig = _messages.MessageField('GooglePrivacyDlpV2ReplaceValueConfig', 9)
    replaceDictionaryConfig = _messages.MessageField('GooglePrivacyDlpV2ReplaceDictionaryConfig', 10)
    replaceWithInfoTypeConfig = _messages.MessageField('GooglePrivacyDlpV2ReplaceWithInfoTypeConfig', 11)
    timePartConfig = _messages.MessageField('GooglePrivacyDlpV2TimePartConfig', 12)