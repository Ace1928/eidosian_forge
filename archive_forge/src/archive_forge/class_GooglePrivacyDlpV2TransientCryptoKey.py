from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GooglePrivacyDlpV2TransientCryptoKey(_messages.Message):
    """Use this to have a random data crypto key generated. It will be
  discarded after the request finishes.

  Fields:
    name: Required. Name of the key. This is an arbitrary string used to
      differentiate different keys. A unique key is generated per name: two
      separate `TransientCryptoKey` protos share the same generated key if
      their names are the same. When the data crypto key is generated, this
      name is not used in any way (repeating the api call will result in a
      different key being generated).
  """
    name = _messages.StringField(1)