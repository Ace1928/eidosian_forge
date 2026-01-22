from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsGetRequest(_messages.Message):
    """A
  ArtifactregistryProjectsLocationsRepositoriesPackagesVersionsGetRequest
  object.

  Enums:
    ViewValueValuesEnum: The view that should be returned in the response.

  Fields:
    name: The name of the version to retrieve.
    view: The view that should be returned in the response.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """The view that should be returned in the response.

    Values:
      VERSION_VIEW_UNSPECIFIED: The default / unset value. The API will
        default to the BASIC view.
      BASIC: Includes basic information about the version, but not any related
        tags.
      FULL: Include everything.
    """
        VERSION_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)