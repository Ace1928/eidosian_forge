from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1SearchCatalogRequestScope(_messages.Message):
    """The criteria that select the subspace used for query matching.

  Fields:
    includeGcpPublicDatasets: If `true`, include Google Cloud public datasets
      in the search results. Info on Google Cloud public datasets is available
      at https://cloud.google.com/public-datasets/. By default, Google Cloud
      public datasets are excluded.
    includeOrgIds: The list of organization IDs to search within. To find your
      organization ID, follow instructions in
      https://cloud.google.com/resource-manager/docs/creating-managing-
      organization.
    includeProjectIds: The list of project IDs to search within. To learn more
      about the distinction between project names/IDs/numbers, go to
      https://cloud.google.com/docs/overview/#projects.
    restrictedLocations: Optional. The list of locations to search within. 1.
      If empty, search will be performed in all locations; 2. If any of the
      locations are NOT in the valid locations list, error will be returned;
      3. Otherwise, search only the given locations for matching results.
      Typical usage is to leave this field empty. When a location is
      unreachable as returned in the `SearchCatalogResponse.unreachable`
      field, users can repeat the search request with this parameter set to
      get additional information on the error. Valid locations: * asia-east1 *
      asia-east2 * asia-northeast1 * asia-northeast2 * asia-northeast3 * asia-
      south1 * asia-southeast1 * australia-southeast1 * eu * europe-north1 *
      europe-west1 * europe-west2 * europe-west3 * europe-west4 * europe-west6
      * global * northamerica-northeast1 * southamerica-east1 * us * us-
      central1 * us-east1 * us-east4 * us-west1 * us-west2
  """
    includeGcpPublicDatasets = _messages.BooleanField(1)
    includeOrgIds = _messages.StringField(2, repeated=True)
    includeProjectIds = _messages.StringField(3, repeated=True)
    restrictedLocations = _messages.StringField(4, repeated=True)