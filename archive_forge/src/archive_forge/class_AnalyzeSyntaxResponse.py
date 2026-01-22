from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnalyzeSyntaxResponse(_messages.Message):
    """The syntax analysis response message.

  Fields:
    language: The language of the text, which will be the same as the language
      specified in the request or, if not specified, the automatically-
      detected language. See Document.language field for more details.
    sentences: Sentences in the input document.
    tokens: Tokens, along with their syntactic information, in the input
      document.
  """
    language = _messages.StringField(1)
    sentences = _messages.MessageField('Sentence', 2, repeated=True)
    tokens = _messages.MessageField('Token', 3, repeated=True)