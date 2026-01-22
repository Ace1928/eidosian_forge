from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1ExamplesOverride(_messages.Message):
    """Overrides for example-based explanations.

  Enums:
    DataFormatValueValuesEnum: The format of the data being provided with each
      call.

  Fields:
    crowdingCount: The number of neighbors to return that have the same
      crowding tag.
    dataFormat: The format of the data being provided with each call.
    neighborCount: The number of neighbors to return.
    restrictions: Restrict the resulting nearest neighbors to respect these
      constraints.
    returnEmbeddings: If true, return the embeddings instead of neighbors.
  """

    class DataFormatValueValuesEnum(_messages.Enum):
        """The format of the data being provided with each call.

    Values:
      DATA_FORMAT_UNSPECIFIED: Unspecified format. Must not be used.
      INSTANCES: Provided data is a set of model inputs.
      EMBEDDINGS: Provided data is a set of embeddings.
    """
        DATA_FORMAT_UNSPECIFIED = 0
        INSTANCES = 1
        EMBEDDINGS = 2
    crowdingCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    dataFormat = _messages.EnumField('DataFormatValueValuesEnum', 2)
    neighborCount = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    restrictions = _messages.MessageField('GoogleCloudAiplatformV1beta1ExamplesRestrictionsNamespace', 4, repeated=True)
    returnEmbeddings = _messages.BooleanField(5)