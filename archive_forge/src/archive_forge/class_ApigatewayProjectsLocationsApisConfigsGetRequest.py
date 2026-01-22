from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigatewayProjectsLocationsApisConfigsGetRequest(_messages.Message):
    """A ApigatewayProjectsLocationsApisConfigsGetRequest object.

  Enums:
    ViewValueValuesEnum: Specifies which fields of the API Config are returned
      in the response. Defaults to `BASIC` view.

  Fields:
    name: Required. Resource name of the form:
      `projects/*/locations/global/apis/*/configs/*`
    view: Specifies which fields of the API Config are returned in the
      response. Defaults to `BASIC` view.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Specifies which fields of the API Config are returned in the response.
    Defaults to `BASIC` view.

    Values:
      CONFIG_VIEW_UNSPECIFIED: <no description>
      BASIC: Do not include configuration source files.
      FULL: Include configuration source files.
    """
        CONFIG_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)