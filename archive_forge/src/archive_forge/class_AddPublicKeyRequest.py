from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AddPublicKeyRequest(_messages.Message):
    """Request message for AddPublicKey.

  Fields:
    key: Key that should be added to the environment. Supported formats are
      `ssh-dss` (see RFC4253), `ssh-rsa` (see RFC4253), `ecdsa-sha2-nistp256`
      (see RFC5656), `ecdsa-sha2-nistp384` (see RFC5656) and `ecdsa-
      sha2-nistp521` (see RFC5656). It should be structured as <format>
      <content>, where <content> part is encoded with Base64.
  """
    key = _messages.StringField(1)