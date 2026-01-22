from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudAiNlLlmProtoServiceRaiResult(_messages.Message):
    """The RAI results for a given text.

  Fields:
    aidaRecitationResult: Recitation result from Aida recitation checker.
    blocked: Use `triggered_blocklist`.
    errorCodes: The error codes indicate which RAI filters block the response.
    filtered: Whether the text should be filtered and not shown to the end
      user. This is determined based on a combination of
      `triggered_recitation`, `triggered_blocklist`, `language_filter_result`,
      and `triggered_safety_filter`.
    languageFilterResult: Language filter result from SAFT LangId.
    raiSignals: The RAI signals for the text.
    translationRequestInfos: Translation request info during RAI for debugging
      purpose. Each TranslationRequestInfo corresponds to a request sent to
      the translation server.
    triggeredBlocklist: Whether the text triggered the blocklist.
    triggeredRecitation: Whether the text should be blocked by the recitation
      result from Aida recitation checker. It is determined from
      aida_recitation_result.
    triggeredSafetyFilter: Whether the text triggered the safety filter.
      Currently, this is due to CSAI triggering or one of four categories
      (derogatory, sexual, toxic, violent) having a score over the filter
      threshold.
  """
    aidaRecitationResult = _messages.MessageField('LanguageLabsAidaTrustRecitationProtoRecitationResult', 1)
    blocked = _messages.BooleanField(2)
    errorCodes = _messages.IntegerField(3, repeated=True, variant=_messages.Variant.INT32)
    filtered = _messages.BooleanField(4)
    languageFilterResult = _messages.MessageField('LearningGenaiRootLanguageFilterResult', 5)
    raiSignals = _messages.MessageField('CloudAiNlLlmProtoServiceRaiSignal', 6, repeated=True)
    translationRequestInfos = _messages.MessageField('LearningGenaiRootTranslationRequestInfo', 7, repeated=True)
    triggeredBlocklist = _messages.BooleanField(8)
    triggeredRecitation = _messages.BooleanField(9)
    triggeredSafetyFilter = _messages.BooleanField(10)