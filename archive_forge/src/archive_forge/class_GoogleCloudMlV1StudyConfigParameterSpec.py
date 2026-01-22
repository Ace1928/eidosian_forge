from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1StudyConfigParameterSpec(_messages.Message):
    """Represents a single parameter to optimize.

  Enums:
    ScaleTypeValueValuesEnum: How the parameter should be scaled. Leave unset
      for categorical parameters.
    TypeValueValuesEnum: Required. The type of the parameter.

  Fields:
    categoricalValueSpec: The value spec for a 'CATEGORICAL' parameter.
    childParameterSpecs: A child node is active if the parameter's value
      matches the child node's matching_parent_values. If two items in
      child_parameter_specs have the same name, they must have disjoint
      matching_parent_values.
    discreteValueSpec: The value spec for a 'DISCRETE' parameter.
    doubleValueSpec: The value spec for a 'DOUBLE' parameter.
    integerValueSpec: The value spec for an 'INTEGER' parameter.
    parameter: Required. The parameter name must be unique amongst all
      ParameterSpecs.
    parentCategoricalValues: A
      GoogleCloudMlV1StudyConfigParameterSpecMatchingParentCategoricalValueSpe
      c attribute.
    parentDiscreteValues: A
      GoogleCloudMlV1StudyConfigParameterSpecMatchingParentDiscreteValueSpec
      attribute.
    parentIntValues: A
      GoogleCloudMlV1StudyConfigParameterSpecMatchingParentIntValueSpec
      attribute.
    scaleType: How the parameter should be scaled. Leave unset for categorical
      parameters.
    type: Required. The type of the parameter.
  """

    class ScaleTypeValueValuesEnum(_messages.Enum):
        """How the parameter should be scaled. Leave unset for categorical
    parameters.

    Values:
      SCALE_TYPE_UNSPECIFIED: By default, no scaling is applied.
      UNIT_LINEAR_SCALE: Scales the feasible space to (0, 1) linearly.
      UNIT_LOG_SCALE: Scales the feasible space logarithmically to (0, 1). The
        entire feasible space must be strictly positive.
      UNIT_REVERSE_LOG_SCALE: Scales the feasible space "reverse"
        logarithmically to (0, 1). The result is that values close to the top
        of the feasible space are spread out more than points near the bottom.
        The entire feasible space must be strictly positive.
    """
        SCALE_TYPE_UNSPECIFIED = 0
        UNIT_LINEAR_SCALE = 1
        UNIT_LOG_SCALE = 2
        UNIT_REVERSE_LOG_SCALE = 3

    class TypeValueValuesEnum(_messages.Enum):
        """Required. The type of the parameter.

    Values:
      PARAMETER_TYPE_UNSPECIFIED: You must specify a valid type. Using this
        unspecified type will result in an error.
      DOUBLE: Type for real-valued parameters.
      INTEGER: Type for integral parameters.
      CATEGORICAL: The parameter is categorical, with a value chosen from the
        categories field.
      DISCRETE: The parameter is real valued, with a fixed set of feasible
        points. If `type==DISCRETE`, feasible_points must be provided, and
        {`min_value`, `max_value`} will be ignored.
    """
        PARAMETER_TYPE_UNSPECIFIED = 0
        DOUBLE = 1
        INTEGER = 2
        CATEGORICAL = 3
        DISCRETE = 4
    categoricalValueSpec = _messages.MessageField('GoogleCloudMlV1StudyConfigParameterSpecCategoricalValueSpec', 1)
    childParameterSpecs = _messages.MessageField('GoogleCloudMlV1StudyConfigParameterSpec', 2, repeated=True)
    discreteValueSpec = _messages.MessageField('GoogleCloudMlV1StudyConfigParameterSpecDiscreteValueSpec', 3)
    doubleValueSpec = _messages.MessageField('GoogleCloudMlV1StudyConfigParameterSpecDoubleValueSpec', 4)
    integerValueSpec = _messages.MessageField('GoogleCloudMlV1StudyConfigParameterSpecIntegerValueSpec', 5)
    parameter = _messages.StringField(6)
    parentCategoricalValues = _messages.MessageField('GoogleCloudMlV1StudyConfigParameterSpecMatchingParentCategoricalValueSpec', 7)
    parentDiscreteValues = _messages.MessageField('GoogleCloudMlV1StudyConfigParameterSpecMatchingParentDiscreteValueSpec', 8)
    parentIntValues = _messages.MessageField('GoogleCloudMlV1StudyConfigParameterSpecMatchingParentIntValueSpec', 9)
    scaleType = _messages.EnumField('ScaleTypeValueValuesEnum', 10)
    type = _messages.EnumField('TypeValueValuesEnum', 11)