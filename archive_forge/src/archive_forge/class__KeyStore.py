from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import collections
import enum
import hashlib
import re
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.cache import function_result_cache
class _KeyStore:
    """Holds encryption key information.

  Attributes:
    encryption_key (Optional[EncryptionKey]): The key for encryption.
    decryption_key_index (Dict[EncryptionKey.sha256, EncryptionKey]): Indexes
      keys that can be used for decryption.
    initialized (bool): True if encryption_key and decryption_key_index
      reflect the values they should based on flag and key file values.
  """

    def __init__(self, encryption_key=None, decryption_key_index=None, initialized=False):
        self.encryption_key = encryption_key
        self.decryption_key_index = decryption_key_index or {}
        self.initialized = initialized

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.encryption_key == other.encryption_key and self.decryption_key_index == other.decryption_key_index and (self.initialized == other.initialized)