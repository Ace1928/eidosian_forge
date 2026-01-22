from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppconfigmanagerProjectsLocationsConfigsVersionRendersGetRequest(_messages.Message):
    """A AppconfigmanagerProjectsLocationsConfigsVersionRendersGetRequest
  object.

  Enums:
    ViewValueValuesEnum: Optional. View of the ConfigVersionRender. In the
      default BASIC view, only the metadata associated with the
      ConfigVersionRender will be returned.

  Fields:
    name: Required. Name of the resource
    view: Optional. View of the ConfigVersionRender. In the default BASIC
      view, only the metadata associated with the ConfigVersionRender will be
      returned.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Optional. View of the ConfigVersionRender. In the default BASIC view,
    only the metadata associated with the ConfigVersionRender will be
    returned.

    Values:
      VIEW_UNSPECIFIED: The default / unset value. The API will default to the
        BASIC view for LIST calls & FULL for GET calls..
      BASIC: Include only the metadata for the resource. This is the default
        view.
      FULL: Include metadata & other relevant payload data as well. For a
        ConfigVersion this implies that the response will hold the user
        provided payload. For a ConfigVersionRender this implies that the
        response will hold the user provided payload along with the rendered
        payload data.
    """
        VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)