from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CleanTextField(_messages.Message):
    """Inspect text and transform sensitive text. Configure using TextConfig.
  Supported [types](https://www.hl7.org/fhir/datatypes.html): Code, Date,
  DateTime, Decimal, HumanName, Id, LanguageCode, Markdown, Oid, String, Uri,
  Uuid, Xhtml.
  """