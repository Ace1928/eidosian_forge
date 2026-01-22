from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MonitoringServicesServiceLevelObjectivesListRequest(_messages.Message):
    """A MonitoringServicesServiceLevelObjectivesListRequest object.

  Enums:
    ViewValueValuesEnum: View of the ServiceLevelObjectives to return. If
      DEFAULT, return each ServiceLevelObjective as originally defined. If
      EXPLICIT and the ServiceLevelObjective is defined in terms of a
      BasicSli, replace the BasicSli with a RequestBasedSli spelling out how
      the SLI is computed.

  Fields:
    filter: A filter specifying what ServiceLevelObjectives to return.
    pageSize: A non-negative number that is the maximum number of results to
      return. When 0, use default page size.
    pageToken: If this field is not empty then it must contain the
      nextPageToken value returned by a previous call to this method. Using
      this field causes the method to return additional results from the
      previous method call.
    parent: Required. Resource name of the parent containing the listed SLOs,
      either a project or a Monitoring Metrics Scope. The formats are:
      projects/[PROJECT_ID_OR_NUMBER]/services/[SERVICE_ID]
      workspaces/[HOST_PROJECT_ID_OR_NUMBER]/services/-
    view: View of the ServiceLevelObjectives to return. If DEFAULT, return
      each ServiceLevelObjective as originally defined. If EXPLICIT and the
      ServiceLevelObjective is defined in terms of a BasicSli, replace the
      BasicSli with a RequestBasedSli spelling out how the SLI is computed.
  """

    class ViewValueValuesEnum(_messages.Enum):
        """View of the ServiceLevelObjectives to return. If DEFAULT, return each
    ServiceLevelObjective as originally defined. If EXPLICIT and the
    ServiceLevelObjective is defined in terms of a BasicSli, replace the
    BasicSli with a RequestBasedSli spelling out how the SLI is computed.

    Values:
      VIEW_UNSPECIFIED: Same as FULL.
      FULL: Return the embedded ServiceLevelIndicator in the form in which it
        was defined. If it was defined using a BasicSli, return that BasicSli.
      EXPLICIT: For ServiceLevelIndicators using BasicSli articulation,
        instead return the ServiceLevelIndicator with its mode of computation
        fully spelled out as a RequestBasedSli. For ServiceLevelIndicators
        using RequestBasedSli or WindowsBasedSli, return the
        ServiceLevelIndicator as it was provided.
    """
        VIEW_UNSPECIFIED = 0
        FULL = 1
        EXPLICIT = 2
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    view = _messages.EnumField('ViewValueValuesEnum', 5)