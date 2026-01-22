from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class KeySourceValueValuesEnum(_messages.Enum):
    """Immutable. The source of the voting key for the blockchain validator.

    Values:
      KEY_SOURCE_UNSPECIFIED: Voting key source has not been specified, but
        should be.
      REMOTE_WEB3_SIGNER: The voting key is stored in a remote signing service
        (Web3Signer) and signing requests are delegated.
      SEED_PHRASE_REFERENCE: Derive voting keys from new seed material.
      EXISTING_SEED_PHRASE_REFERENCE: Derive voting keys from existing seed
        material.
    """
    KEY_SOURCE_UNSPECIFIED = 0
    REMOTE_WEB3_SIGNER = 1
    SEED_PHRASE_REFERENCE = 2
    EXISTING_SEED_PHRASE_REFERENCE = 3