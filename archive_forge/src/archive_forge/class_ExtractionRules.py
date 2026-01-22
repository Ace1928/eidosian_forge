from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExtractionRules(_messages.Message):
    """Extraction Rules to identity the backends from customer provided
  configuration in Connection resource.

  Fields:
    extractionRule: Collection of Extraction Rule.
  """
    extractionRule = _messages.MessageField('ExtractionRule', 1, repeated=True)