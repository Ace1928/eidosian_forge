from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1SchemaTrainingjobDefinitionAutoMlTablesInputsTransformationCategoricalArrayTransformation(_messages.Message):
    """Treats the column as categorical array and performs following
  transformation functions. * For each element in the array, convert the
  category name to a dictionary lookup index and generate an embedding for
  each index. Combine the embedding of all elements into a single embedding
  using the mean. * Empty arrays treated as an embedding of zeroes.

  Fields:
    columnName: A string attribute.
  """
    columnName = _messages.StringField(1)