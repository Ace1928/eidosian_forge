from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ConnectorsProjectsLocationsProvidersConnectorsVersionsListRequest(_messages.Message):
    """A ConnectorsProjectsLocationsProvidersConnectorsVersionsListRequest
  object.

  Enums:
    ViewValueValuesEnum: Specifies which fields of the ConnectorVersion are
      returned in the response. Defaults to `BASIC` view.

  Fields:
    pageSize: Page size.
    pageToken: Page token.
    parent: Required. Parent resource of the connectors, of the form:
      `projects/*/locations/*/providers/*/connectors/*` Only global location
      is supported for ConnectorVersion resource.
    view: Specifies which fields of the ConnectorVersion are returned in the
      response. Defaults to `BASIC` view.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Specifies which fields of the ConnectorVersion are returned in the
    response. Defaults to `BASIC` view.

    Values:
      CONNECTOR_VERSION_VIEW_UNSPECIFIED: CONNECTOR_VERSION_VIEW_UNSPECIFIED.
      CONNECTOR_VERSION_VIEW_BASIC: Do not include role grant configs.
      CONNECTOR_VERSION_VIEW_FULL: Include role grant configs.
    """
        CONNECTOR_VERSION_VIEW_UNSPECIFIED = 0
        CONNECTOR_VERSION_VIEW_BASIC = 1
        CONNECTOR_VERSION_VIEW_FULL = 2
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 4)