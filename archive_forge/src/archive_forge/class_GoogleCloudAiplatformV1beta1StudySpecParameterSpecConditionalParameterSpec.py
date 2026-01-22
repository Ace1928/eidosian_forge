from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAiplatformV1beta1StudySpecParameterSpecConditionalParameterSpec(_messages.Message):
    """Represents a parameter spec with condition from its parent parameter.

  Fields:
    parameterSpec: Required. The spec for a conditional parameter.
    parentCategoricalValues: The spec for matching values from a parent
      parameter of `CATEGORICAL` type.
    parentDiscreteValues: The spec for matching values from a parent parameter
      of `DISCRETE` type.
    parentIntValues: The spec for matching values from a parent parameter of
      `INTEGER` type.
  """
    parameterSpec = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecParameterSpec', 1)
    parentCategoricalValues = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecParameterSpecConditionalParameterSpecCategoricalValueCondition', 2)
    parentDiscreteValues = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecParameterSpecConditionalParameterSpecDiscreteValueCondition', 3)
    parentIntValues = _messages.MessageField('GoogleCloudAiplatformV1beta1StudySpecParameterSpecConditionalParameterSpecIntValueCondition', 4)