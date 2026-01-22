from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BeyondcorpOrganizationsLocationsInsightsGetRequest(_messages.Message):
    """A BeyondcorpOrganizationsLocationsInsightsGetRequest object.

  Enums:
    ViewValueValuesEnum: Required. Metadata only or full data view.

  Fields:
    name: Required. The resource name of the insight using the form: `organiza
      tions/{organization_id}/locations/{location_id}/insights/{insight_id}`
      `projects/{project_id}/locations/{location_id}/insights/{insight_id}`
    view: Required. Metadata only or full data view.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """Required. Metadata only or full data view.

    Values:
      INSIGHT_VIEW_UNSPECIFIED: The default / unset value. The API will
        default to the BASIC view.
      BASIC: Include basic metadata about the insight, but not the insight
        data. This is the default value (for both ListInsights and
        GetInsight).
      FULL: Include everything.
    """
        INSIGHT_VIEW_UNSPECIFIED = 0
        BASIC = 1
        FULL = 2
    name = _messages.StringField(1, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 2)