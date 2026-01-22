from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AssuredworkloadsOrganizationsLocationsWorkloadsAnalyzeWorkloadMoveRequest(_messages.Message):
    """A
  AssuredworkloadsOrganizationsLocationsWorkloadsAnalyzeWorkloadMoveRequest
  object.

  Fields:
    assetTypes: Optional. List of asset types to be analyzed, including and
      under the source resource. If empty, all assets are analyzed. The
      complete list of asset types is available
      [here](https://cloud.google.com/asset-inventory/docs/supported-asset-
      types).
    pageSize: Optional. Page size. If a value is not specified, the default
      value of 10 is used.
    pageToken: Optional. The page token from the previous response. It needs
      to be passed in the second and following requests.
    project: The source type is a project. Specify the project's relative
      resource name, formatted as either a project number or a project ID:
      "projects/{PROJECT_NUMBER}" or "projects/{PROJECT_ID}" For example:
      "projects/951040570662" when specifying a project number, or
      "projects/my-project-123" when specifying a project ID.
    target: Required. The resource ID of the folder-based destination
      workload. This workload is where the source resource will hypothetically
      be moved to. Specify the workload's relative resource name, formatted
      as: "organizations/{ORGANIZATION_ID}/locations/{LOCATION_ID}/workloads/{
      WORKLOAD_ID}" For example: "organizations/123/locations/us-
      east1/workloads/assured-workload-2"
  """
    assetTypes = _messages.StringField(1, repeated=True)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    project = _messages.StringField(4)
    target = _messages.StringField(5, required=True)