from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SpeechProjectsLocationsPhraseSetsListRequest(_messages.Message):
    """A SpeechProjectsLocationsPhraseSetsListRequest object.

  Fields:
    pageSize: The maximum number of PhraseSets to return. The service may
      return fewer than this value. If unspecified, at most 5 PhraseSets will
      be returned. The maximum value is 100; values above 100 will be coerced
      to 100.
    pageToken: A page token, received from a previous ListPhraseSets call.
      Provide this to retrieve the subsequent page. When paginating, all other
      parameters provided to ListPhraseSets must match the call that provided
      the page token.
    parent: Required. The project and location of PhraseSet resources to list.
      The expected format is `projects/{project}/locations/{location}`.
    showDeleted: Whether, or not, to show resources that have been deleted.
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    showDeleted = _messages.BooleanField(4)