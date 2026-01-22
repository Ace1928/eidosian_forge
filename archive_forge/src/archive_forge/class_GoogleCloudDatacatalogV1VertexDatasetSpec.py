from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1VertexDatasetSpec(_messages.Message):
    """Specification for vertex dataset resources.

  Enums:
    DataTypeValueValuesEnum: Type of the dataset.

  Fields:
    dataItemCount: The number of DataItems in this Dataset. Only apply for
      non-structured Dataset.
    dataType: Type of the dataset.
  """

    class DataTypeValueValuesEnum(_messages.Enum):
        """Type of the dataset.

    Values:
      DATA_TYPE_UNSPECIFIED: Should not be used.
      TABLE: Structured data dataset.
      IMAGE: Image dataset which supports ImageClassification,
        ImageObjectDetection and ImageSegmentation problems.
      TEXT: Document dataset which supports TextClassification, TextExtraction
        and TextSentiment problems.
      VIDEO: Video dataset which supports VideoClassification,
        VideoObjectTracking and VideoActionRecognition problems.
      CONVERSATION: Conversation dataset which supports conversation problems.
      TIME_SERIES: TimeSeries dataset.
      DOCUMENT: Document dataset which supports DocumentAnnotation problems.
      TEXT_TO_SPEECH: TextToSpeech dataset which supports TextToSpeech
        problems.
      TRANSLATION: Translation dataset which supports Translation problems.
      STORE_VISION: Store Vision dataset which is used for HITL integration.
      ENTERPRISE_KNOWLEDGE_GRAPH: Enterprise Knowledge Graph dataset which is
        used for HITL labeling integration.
      TEXT_PROMPT: Text prompt dataset which supports Large Language Models.
    """
        DATA_TYPE_UNSPECIFIED = 0
        TABLE = 1
        IMAGE = 2
        TEXT = 3
        VIDEO = 4
        CONVERSATION = 5
        TIME_SERIES = 6
        DOCUMENT = 7
        TEXT_TO_SPEECH = 8
        TRANSLATION = 9
        STORE_VISION = 10
        ENTERPRISE_KNOWLEDGE_GRAPH = 11
        TEXT_PROMPT = 12
    dataItemCount = _messages.IntegerField(1)
    dataType = _messages.EnumField('DataTypeValueValuesEnum', 2)