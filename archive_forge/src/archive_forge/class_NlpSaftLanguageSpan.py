from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NlpSaftLanguageSpan(_messages.Message):
    """A NlpSaftLanguageSpan object.

  Fields:
    end: A integer attribute.
    languageCode: A BCP 47 language code for this span.
    locales: Optional field containing any information that was predicted
      about the specific locale(s) of the span.
    probability: A probability associated with this prediction.
    start: Start and end byte offsets, inclusive, within the given input
      string. A value of -1 implies that this field is not set. Both fields
      must either be set with a nonnegative value or both are unset. If both
      are unset then this LanguageSpan applies to the entire input.
  """
    end = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    languageCode = _messages.StringField(2)
    locales = _messages.MessageField('NlpSaftLangIdLocalesResult', 3)
    probability = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    start = _messages.IntegerField(5, variant=_messages.Variant.INT32)