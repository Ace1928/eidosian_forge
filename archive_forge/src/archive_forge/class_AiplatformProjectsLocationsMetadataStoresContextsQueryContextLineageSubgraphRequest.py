from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsMetadataStoresContextsQueryContextLineageSubgraphRequest(_messages.Message):
    """A AiplatformProjectsLocationsMetadataStoresContextsQueryContextLineageSu
  bgraphRequest object.

  Fields:
    context: Required. The resource name of the Context whose Artifacts and
      Executions should be retrieved as a LineageSubgraph. Format: `projects/{
      project}/locations/{location}/metadataStores/{metadatastore}/contexts/{c
      ontext}` The request may error with FAILED_PRECONDITION if the number of
      Artifacts, the number of Executions, or the number of Events that would
      be returned for the Context exceeds 1000.
  """
    context = _messages.StringField(1, required=True)