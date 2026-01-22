from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExtractionRule(_messages.Message):
    """Extraction Rule.

  Fields:
    extractionRegex: Regex used to extract backend details from source. If
      empty, whole source value will be used.
    source: Source on which the rule is applied.
  """
    extractionRegex = _messages.StringField(1)
    source = _messages.MessageField('Source', 2)