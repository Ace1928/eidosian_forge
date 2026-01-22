from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsConnectionsRepositoriesFetchGitRefsRequest(_messages.Message):
    """A CloudbuildProjectsLocationsConnectionsRepositoriesFetchGitRefsRequest
  object.

  Enums:
    RefTypeValueValuesEnum: Type of refs to fetch

  Fields:
    pageSize: Optional. Number of results to return in the list. Default to
      20.
    pageToken: Optional. Page start.
    refType: Type of refs to fetch
    repository: Required. The resource name of the repository in the format
      `projects/*/locations/*/connections/*/repositories/*`.
  """

    class RefTypeValueValuesEnum(_messages.Enum):
        """Type of refs to fetch

    Values:
      REF_TYPE_UNSPECIFIED: No type specified.
      TAG: To fetch tags.
      BRANCH: To fetch branches.
    """
        REF_TYPE_UNSPECIFIED = 0
        TAG = 1
        BRANCH = 2
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    refType = _messages.EnumField('RefTypeValueValuesEnum', 3)
    repository = _messages.StringField(4, required=True)