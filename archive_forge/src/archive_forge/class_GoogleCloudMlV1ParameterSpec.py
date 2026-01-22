from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1ParameterSpec(_messages.Message):
    """Represents a single hyperparameter to optimize.

  Enums:
    ScaleTypeValueValuesEnum: Optional. How the parameter should be scaled to
      the hypercube. Leave unset for categorical parameters. Some kind of
      scaling is strongly recommended for real or integral parameters (e.g.,
      `UNIT_LINEAR_SCALE`).
    TypeValueValuesEnum: Required. The type of the parameter.

  Fields:
    categoricalValues: Required if type is `CATEGORICAL`. The list of possible
      categories.
    discreteValues: Required if type is `DISCRETE`. A list of feasible points.
      The list should be in strictly increasing order. For instance, this
      parameter might have possible settings of 1.5, 2.5, and 4.0. This list
      should not contain more than 1,000 values.
    maxValue: Required if type is `DOUBLE` or `INTEGER`. This field should be
      unset if type is `CATEGORICAL`. This value should be integers if type is
      `INTEGER`.
    minValue: Required if type is `DOUBLE` or `INTEGER`. This field should be
      unset if type is `CATEGORICAL`. This value should be integers if type is
      INTEGER.
    parameterName: Required. The parameter name must be unique amongst all
      ParameterConfigs in a HyperparameterSpec message. E.g., "learning_rate".
    scaleType: Optional. How the parameter should be scaled to the hypercube.
      Leave unset for categorical parameters. Some kind of scaling is strongly
      recommended for real or integral parameters (e.g., `UNIT_LINEAR_SCALE`).
    type: Required. The type of the parameter.
  """

    class ScaleTypeValueValuesEnum(_messages.Enum):
        """Optional. How the parameter should be scaled to the hypercube. Leave
    unset for categorical parameters. Some kind of scaling is strongly
    recommended for real or integral parameters (e.g., `UNIT_LINEAR_SCALE`).

    Values:
      NONE: By default, no scaling is applied.
      UNIT_LINEAR_SCALE: Scales the feasible space to (0, 1) linearly.
      UNIT_LOG_SCALE: Scales the feasible space logarithmically to (0, 1). The
        entire feasible space must be strictly positive.
      UNIT_REVERSE_LOG_SCALE: Scales the feasible space "reverse"
        logarithmically to (0, 1). The result is that values close to the top
        of the feasible space are spread out more than points near the bottom.
        The entire feasible space must be strictly positive.
    """
        NONE = 0
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
    categoricalValues = _messages.StringField(1, repeated=True)
    discreteValues = _messages.FloatField(2, repeated=True)
    maxValue = _messages.FloatField(3)
    minValue = _messages.FloatField(4)
    parameterName = _messages.StringField(5)
    scaleType = _messages.EnumField('ScaleTypeValueValuesEnum', 6)
    type = _messages.EnumField('TypeValueValuesEnum', 7)