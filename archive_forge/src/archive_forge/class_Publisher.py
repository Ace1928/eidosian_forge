from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Publisher(_messages.Message):
    """Publisher contains information about the publisher of this Note.

  Fields:
    issuingAuthority: Provides information about the authority of the issuing
      party to release the document, in particular, the party's constituency
      and responsibilities or other obligations.
    name: Name of the publisher. Examples: 'Google', 'Google Cloud Platform'.
    publisherNamespace: The context or namespace. Contains a URL which is
      under control of the issuing party and can be used as a globally unique
      identifier for that issuing party. Example: https://csaf.io
  """
    issuingAuthority = _messages.StringField(1)
    name = _messages.StringField(2)
    publisherNamespace = _messages.StringField(3)