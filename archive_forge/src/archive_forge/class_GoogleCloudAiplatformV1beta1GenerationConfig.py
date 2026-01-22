from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1GenerationConfig(_messages.Message):
    """Generation config.

  Fields:
    candidateCount: Optional. Number of candidates to generate.
    frequencyPenalty: Optional. Frequency penalties.
    maxOutputTokens: Optional. The maximum number of output tokens to generate
      per message.
    presencePenalty: Optional. Positive penalties.
    responseMimeType: Optional. Output response mimetype of the generated
      candidate text. Supported mimetype: - `text/plain`: (default) Text
      output. - `application/json`: JSON response in the candidates. The model
      needs to be prompted to output the appropriate response type, otherwise
      the behavior is undefined. This is a preview feature.
    stopSequences: Optional. Stop sequences.
    temperature: Optional. Controls the randomness of predictions.
    topK: Optional. If specified, top-k sampling will be used.
    topP: Optional. If specified, nucleus sampling will be used.
  """
    candidateCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    frequencyPenalty = _messages.FloatField(2, variant=_messages.Variant.FLOAT)
    maxOutputTokens = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    presencePenalty = _messages.FloatField(4, variant=_messages.Variant.FLOAT)
    responseMimeType = _messages.StringField(5)
    stopSequences = _messages.StringField(6, repeated=True)
    temperature = _messages.FloatField(7, variant=_messages.Variant.FLOAT)
    topK = _messages.FloatField(8, variant=_messages.Variant.FLOAT)
    topP = _messages.FloatField(9, variant=_messages.Variant.FLOAT)