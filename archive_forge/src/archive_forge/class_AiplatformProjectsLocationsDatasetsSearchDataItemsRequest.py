from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AiplatformProjectsLocationsDatasetsSearchDataItemsRequest(_messages.Message):
    """A AiplatformProjectsLocationsDatasetsSearchDataItemsRequest object.

  Fields:
    annotationFilters: An expression that specifies what Annotations will be
      returned per DataItem. Annotations satisfied either of the conditions
      will be returned. * `annotation_spec_id` - for = or !=. Must specify
      `saved_query_id=` - saved query id that annotations should belong to.
    annotationsFilter: An expression for filtering the Annotations that will
      be returned per DataItem. * `annotation_spec_id` - for = or !=.
    annotationsLimit: If set, only up to this many of Annotations will be
      returned per DataItemView. The maximum value is 1000. If not set, the
      maximum value will be used.
    dataItemFilter: An expression for filtering the DataItem that will be
      returned. * `data_item_id` - for = or !=. * `labeled` - for = or !=. *
      `has_annotation(ANNOTATION_SPEC_ID)` - true only for DataItem that have
      at least one annotation with annotation_spec_id = `ANNOTATION_SPEC_ID`
      in the context of SavedQuery or DataLabelingJob. For example: *
      `data_item=1` * `has_annotation(5)`
    dataLabelingJob: The resource name of a DataLabelingJob. Format: `projects
      /{project}/locations/{location}/dataLabelingJobs/{data_labeling_job}` If
      this field is set, all of the search will be done in the context of this
      DataLabelingJob.
    dataset: Required. The resource name of the Dataset from which to search
      DataItems. Format:
      `projects/{project}/locations/{location}/datasets/{dataset}`
    fieldMask: Mask specifying which fields of DataItemView to read.
    orderBy: A comma-separated list of fields to order by, sorted in ascending
      order. Use "desc" after a field name for descending.
    orderByAnnotation_orderBy: A comma-separated list of annotation fields to
      order by, sorted in ascending order. Use "desc" after a field name for
      descending. Must also specify saved_query.
    orderByAnnotation_savedQuery: Required. Saved query of the Annotation.
      Only Annotations belong to this saved query will be considered for
      ordering.
    orderByDataItem: A comma-separated list of data item fields to order by,
      sorted in ascending order. Use "desc" after a field name for descending.
    pageSize: Requested page size. Server may return fewer results than
      requested. Default and maximum page size is 100.
    pageToken: A token identifying a page of results for the server to return
      Typically obtained via SearchDataItemsResponse.next_page_token of the
      previous DatasetService.SearchDataItems call.
    savedQuery: The resource name of a SavedQuery(annotation set in UI).
      Format: `projects/{project}/locations/{location}/datasets/{dataset}/save
      dQueries/{saved_query}` All of the search will be done in the context of
      this SavedQuery.
  """
    annotationFilters = _messages.StringField(1, repeated=True)
    annotationsFilter = _messages.StringField(2)
    annotationsLimit = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    dataItemFilter = _messages.StringField(4)
    dataLabelingJob = _messages.StringField(5)
    dataset = _messages.StringField(6, required=True)
    fieldMask = _messages.StringField(7)
    orderBy = _messages.StringField(8)
    orderByAnnotation_orderBy = _messages.StringField(9)
    orderByAnnotation_savedQuery = _messages.StringField(10)
    orderByDataItem = _messages.StringField(11)
    pageSize = _messages.IntegerField(12, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(13)
    savedQuery = _messages.StringField(14)