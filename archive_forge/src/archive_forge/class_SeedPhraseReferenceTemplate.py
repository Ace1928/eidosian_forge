from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SeedPhraseReferenceTemplate(_messages.Message):
    """Configuration for creating voting keys from a new seed phrase, and
  optionally location to back it up to, in Secret Manager.

  Fields:
    exportSeedPhrase: Optional. Immutable. True to export the seed phrase to
      Secret Manager.
    keyCount: Required. Number of keys (and therefore validators) to derive
      from the seed phrase. Must be between 1 and 1,000.
    seedPhraseSecret: Required. Immutable. Reference into Secret Manager for
      where the seed phrase is stored.
  """
    exportSeedPhrase = _messages.BooleanField(1)
    keyCount = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    seedPhraseSecret = _messages.StringField(3)