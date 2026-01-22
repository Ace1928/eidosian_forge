from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtSentencesListRequest(_messages.Message):
    """A
  TranslateProjectsLocationsAdaptiveMtDatasetsAdaptiveMtSentencesListRequest
  object.

  Fields:
    pageSize: A integer attribute.
    pageToken: A token identifying a page of results the server should return.
      Typically, this is the value of
      ListAdaptiveMtSentencesRequest.next_page_token returned from the
      previous call to `ListTranslationMemories` method. The first page is
      returned if `page_token` is empty or missing.
    parent: Required. The resource name of the project from which to list the
      Adaptive MT files. The following format lists all sentences under a
      file. `projects/{project}/locations/{location}/adaptiveMtDatasets/{datas
      et}/adaptiveMtFiles/{file}` The following format lists all sentences
      within a dataset.
      `projects/{project}/locations/{location}/adaptiveMtDatasets/{dataset}`
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)