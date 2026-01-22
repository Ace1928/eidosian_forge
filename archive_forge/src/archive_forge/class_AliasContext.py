from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AliasContext(_messages.Message):
    """An alias to a repo revision.

  Enums:
    KindValueValuesEnum: The alias kind.

  Fields:
    kind: The alias kind.
    name: The alias name.
  """

    class KindValueValuesEnum(_messages.Enum):
        """The alias kind.

    Values:
      KIND_UNSPECIFIED: Unknown.
      FIXED: Git tag.
      MOVABLE: Git branch.
      OTHER: Used to specify non-standard aliases. For example, if a Git repo
        has a ref named "refs/foo/bar".
    """
        KIND_UNSPECIFIED = 0
        FIXED = 1
        MOVABLE = 2
        OTHER = 3
    kind = _messages.EnumField('KindValueValuesEnum', 1)
    name = _messages.StringField(2)