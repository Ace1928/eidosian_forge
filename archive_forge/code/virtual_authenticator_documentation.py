import functools
import typing
from base64 import urlsafe_b64decode
from base64 import urlsafe_b64encode
from enum import Enum
Creates a resident (i.e. stateful) credential.

        :Args:
          - id (bytes): Unique base64 encoded string.
          - rp_id (str): Relying party identifier.
          - user_handle (bytes): userHandle associated to the credential. Must be Base64 encoded string.
          - private_key (bytes): Base64 encoded PKCS
          - sign_count (int): intital value for a signature counter.

        :returns:
          - Credential: A resident credential.
        