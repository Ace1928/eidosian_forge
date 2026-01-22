from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RelationTypeValueValuesEnum(_messages.Enum):
    """The relation between the group and the transitive member.

    Values:
      RELATION_TYPE_UNSPECIFIED: The relation type is undefined or
        undetermined.
      DIRECT: The two entities have only a direct membership with each other.
      INDIRECT: The two entities have only an indirect membership with each
        other.
      DIRECT_AND_INDIRECT: The two entities have both a direct and an indirect
        membership with each other.
    """
    RELATION_TYPE_UNSPECIFIED = 0
    DIRECT = 1
    INDIRECT = 2
    DIRECT_AND_INDIRECT = 3