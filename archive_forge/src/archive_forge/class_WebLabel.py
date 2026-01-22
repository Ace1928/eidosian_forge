from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebLabel(_messages.Message):
    """Label to provide extra metadata for the web detection.

  Fields:
    label: Label for extra metadata.
    languageCode: The BCP-47 language code for `label`, such as "en-US" or
      "sr-Latn". For more information, see
      http://www.unicode.org/reports/tr35/#Unicode_locale_identifier.
  """
    label = _messages.StringField(1)
    languageCode = _messages.StringField(2)