from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CryptoHashField(_messages.Message):
    """Replace field value with a hash of that value. Supported
  [types](https://www.hl7.org/fhir/datatypes.html): Code, Decimal, HumanName,
  Id, LanguageCode, Markdown, Oid, String, Uri, Uuid, Xhtml.
  """