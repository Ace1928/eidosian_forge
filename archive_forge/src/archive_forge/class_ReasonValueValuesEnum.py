from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReasonValueValuesEnum(_messages.Enum):
    """Filter reason.

    Values:
      FILTER_REASON_UNKNOWN: Unknown filter reason.
      FILTER_REASON_NOT_FILTERED: Input not filtered.
      FILTER_REASON_SENSITIVE: Sensitive content.
      FILTER_REASON_RECITATION: Recited content.
      FILTER_REASON_LANGUAGE: Language filtering
      FILTER_REASON_TAKEDOWN: Takedown policy
      FILTER_REASON_CLASSIFIER: Classifier Module
      FILTER_REASON_EMPTY_RESPONSE: Empty response message.
      FILTER_REASON_SIMILARITY_TAKEDOWN: Similarity takedown.
      FILTER_REASON_UNSAFE: Unsafe responses from scorers.
      FILTER_REASON_PAIRWISE_CLASSIFIER: Pairwise classifier.
      FILTER_REASON_CODEY: Codey Filter.
      FILTER_REASON_URL: URLs Filter.
      FILTER_REASON_EMAIL: Emails Filter.
      FILTER_REASON_SAFETY_CAT: SafetyCat filter.
      FILTER_REASON_REQUEST_RESPONSE_TAKEDOWN: Request Response takedown.
      FILTER_REASON_RAI_PQC: RAI Filter.
      FILTER_REASON_ATLAS: Atlas specific topic filter
      FILTER_REASON_RAI_CSAM: RAI Filter.
      FILTER_REASON_RAI_FRINGE: RAI Filter.
      FILTER_REASON_RAI_SPII: RAI Filter.
      FILTER_REASON_RAI_IMAGE_VIOLENCE: RAI Filter
      FILTER_REASON_RAI_IMAGE_PORN: RAI Filter
      FILTER_REASON_RAI_IMAGE_CSAM: RAI Filter
      FILTER_REASON_RAI_IMAGE_PEDO: RAI Filter
      FILTER_REASON_RAI_IMAGE_CHILD: RAI Filter
      FILTER_REASON_RAI_VIDEO_FRAME_VIOLENCE: RAI Filter
      FILTER_REASON_RAI_VIDEO_FRAME_PORN: RAI Filter
      FILTER_REASON_RAI_VIDEO_FRAME_CSAM: RAI Filter
      FILTER_REASON_RAI_VIDEO_FRAME_PEDO: RAI Filter
      FILTER_REASON_RAI_VIDEO_FRAME_CHILD: RAI Filter
      FILTER_REASON_RAI_CONTEXTUAL_DANGEROUS: RAI Filter
      FILTER_REASON_RAI_GRAIL_TEXT: Grail Text
      FILTER_REASON_RAI_GRAIL_IMAGE: Grail Image
      FILTER_REASON_RAI_SAFETYCAT: SafetyCat.
      FILTER_REASON_TOXICITY: Toxic content.
      FILTER_REASON_ATLAS_PRICING: Atlas specific topic filter for pricing
        questions.
      FILTER_REASON_ATLAS_BILLING: Atlas specific topic filter for billing
        questions.
      FILTER_REASON_ATLAS_NON_ENGLISH_QUESTION: Atlas specific topic filter
        for non english questions.
      FILTER_REASON_ATLAS_NOT_RELATED_TO_GCP: Atlas specific topic filter for
        non GCP questions.
      FILTER_REASON_ATLAS_AWS_AZURE_RELATED: Atlas specific topic filter
        aws/azure related questions.
      FILTER_REASON_XAI: Right now we don't do any filtering for XAI. Adding
        this just want to differentiatiat the XAI output metadata from other
        SafetyCat RAI output metadata
      FILTER_CONTROL_DECODING: The response are filtered because it could not
        pass the control decoding thresholds and the maximum rewind attempts
        is reached.
    """
    FILTER_REASON_UNKNOWN = 0
    FILTER_REASON_NOT_FILTERED = 1
    FILTER_REASON_SENSITIVE = 2
    FILTER_REASON_RECITATION = 3
    FILTER_REASON_LANGUAGE = 4
    FILTER_REASON_TAKEDOWN = 5
    FILTER_REASON_CLASSIFIER = 6
    FILTER_REASON_EMPTY_RESPONSE = 7
    FILTER_REASON_SIMILARITY_TAKEDOWN = 8
    FILTER_REASON_UNSAFE = 9
    FILTER_REASON_PAIRWISE_CLASSIFIER = 10
    FILTER_REASON_CODEY = 11
    FILTER_REASON_URL = 12
    FILTER_REASON_EMAIL = 13
    FILTER_REASON_SAFETY_CAT = 14
    FILTER_REASON_REQUEST_RESPONSE_TAKEDOWN = 15
    FILTER_REASON_RAI_PQC = 16
    FILTER_REASON_ATLAS = 17
    FILTER_REASON_RAI_CSAM = 18
    FILTER_REASON_RAI_FRINGE = 19
    FILTER_REASON_RAI_SPII = 20
    FILTER_REASON_RAI_IMAGE_VIOLENCE = 21
    FILTER_REASON_RAI_IMAGE_PORN = 22
    FILTER_REASON_RAI_IMAGE_CSAM = 23
    FILTER_REASON_RAI_IMAGE_PEDO = 24
    FILTER_REASON_RAI_IMAGE_CHILD = 25
    FILTER_REASON_RAI_VIDEO_FRAME_VIOLENCE = 26
    FILTER_REASON_RAI_VIDEO_FRAME_PORN = 27
    FILTER_REASON_RAI_VIDEO_FRAME_CSAM = 28
    FILTER_REASON_RAI_VIDEO_FRAME_PEDO = 29
    FILTER_REASON_RAI_VIDEO_FRAME_CHILD = 30
    FILTER_REASON_RAI_CONTEXTUAL_DANGEROUS = 31
    FILTER_REASON_RAI_GRAIL_TEXT = 32
    FILTER_REASON_RAI_GRAIL_IMAGE = 33
    FILTER_REASON_RAI_SAFETYCAT = 34
    FILTER_REASON_TOXICITY = 35
    FILTER_REASON_ATLAS_PRICING = 36
    FILTER_REASON_ATLAS_BILLING = 37
    FILTER_REASON_ATLAS_NON_ENGLISH_QUESTION = 38
    FILTER_REASON_ATLAS_NOT_RELATED_TO_GCP = 39
    FILTER_REASON_ATLAS_AWS_AZURE_RELATED = 40
    FILTER_REASON_XAI = 41
    FILTER_CONTROL_DECODING = 42