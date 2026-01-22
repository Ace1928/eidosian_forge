from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudDlpDataProfile(_messages.Message):
    """The [data profile](https://cloud.google.com/dlp/docs/data-profiles)
  associated with the finding.

  Enums:
    ParentTypeValueValuesEnum: The resource hierarchy level at which the data
      profile was generated.

  Fields:
    dataProfile: Name of the data profile, for example,
      `projects/123/locations/europe/tableProfiles/8383929`.
    parentType: The resource hierarchy level at which the data profile was
      generated.
  """

    class ParentTypeValueValuesEnum(_messages.Enum):
        """The resource hierarchy level at which the data profile was generated.

    Values:
      PARENT_TYPE_UNSPECIFIED: Unspecified parent type.
      ORGANIZATION: Organization-level configurations.
      PROJECT: Project-level configurations.
    """
        PARENT_TYPE_UNSPECIFIED = 0
        ORGANIZATION = 1
        PROJECT = 2
    dataProfile = _messages.StringField(1)
    parentType = _messages.EnumField('ParentTypeValueValuesEnum', 2)