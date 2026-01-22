from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListTagHoldsResponse(_messages.Message):
    """The ListTagHolds response.

  Fields:
    nextPageToken: Pagination token. If the result set is too large to fit in
      a single response, this token is returned. It encodes the position of
      the current result cursor. Feeding this value into a new list request
      with the `page_token` parameter gives the next page of the results. When
      `next_page_token` is not filled in, there is no next page and the list
      returned is the last page in the result set. Pagination tokens have a
      limited lifetime.
    tagHolds: A possibly paginated list of TagHolds.
  """
    nextPageToken = _messages.StringField(1)
    tagHolds = _messages.MessageField('TagHold', 2, repeated=True)