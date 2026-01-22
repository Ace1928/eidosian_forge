from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataplexProjectsLocationsSearchEntriesRequest(_messages.Message):
    """A DataplexProjectsLocationsSearchEntriesRequest object.

  Fields:
    name: Required. The project to which the request should be attributed in
      the following form: projects/{project}/locations/{location}.
    orderBy: Optional. Ordering of the results. Supported options to be added
      later.
    pageSize: Optional. Pagination.
    pageToken: A string attribute.
    query: Required. The query against which entries in scope should be
      matched.
    scope: Optional. The scope under which the search should be operating.
      Should either be organizations/ or projects/. If left unspecified, it
      will default to the organization where the project provided in name is
      located.
  """
    name = _messages.StringField(1, required=True)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    query = _messages.StringField(5)
    scope = _messages.StringField(6)