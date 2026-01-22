from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DocumentaiProjectsLocationsProcessorTypesListRequest(_messages.Message):
    """A DocumentaiProjectsLocationsProcessorTypesListRequest object.

  Fields:
    pageSize: The maximum number of processor types to return. If unspecified,
      at most `100` processor types will be returned. The maximum value is
      `500`. Values above `500` will be coerced to `500`.
    pageToken: Used to retrieve the next page of results, empty if at the end
      of the list.
    parent: Required. The location of processor types to list. Format:
      `projects/{project}/locations/{location}`.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)