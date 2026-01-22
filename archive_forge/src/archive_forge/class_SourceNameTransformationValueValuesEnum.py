from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SourceNameTransformationValueValuesEnum(_messages.Enum):
    """Optional. Additional transformation that can be done on the source
    entity name before it is being used by the new_name_pattern, for example
    lower case. If no transformation is desired, use NO_TRANSFORMATION

    Values:
      ENTITY_NAME_TRANSFORMATION_UNSPECIFIED: Entity name transformation
        unspecified.
      ENTITY_NAME_TRANSFORMATION_NO_TRANSFORMATION: No transformation.
      ENTITY_NAME_TRANSFORMATION_LOWER_CASE: Transform to lower case.
      ENTITY_NAME_TRANSFORMATION_UPPER_CASE: Transform to upper case.
      ENTITY_NAME_TRANSFORMATION_CAPITALIZED_CASE: Transform to capitalized
        case.
    """
    ENTITY_NAME_TRANSFORMATION_UNSPECIFIED = 0
    ENTITY_NAME_TRANSFORMATION_NO_TRANSFORMATION = 1
    ENTITY_NAME_TRANSFORMATION_LOWER_CASE = 2
    ENTITY_NAME_TRANSFORMATION_UPPER_CASE = 3
    ENTITY_NAME_TRANSFORMATION_CAPITALIZED_CASE = 4