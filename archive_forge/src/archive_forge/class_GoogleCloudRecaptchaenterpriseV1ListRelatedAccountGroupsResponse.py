from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudRecaptchaenterpriseV1ListRelatedAccountGroupsResponse(_messages.Message):
    """The response to a `ListRelatedAccountGroups` call.

  Fields:
    nextPageToken: A token, which can be sent as `page_token` to retrieve the
      next page. If this field is omitted, there are no subsequent pages.
    relatedAccountGroups: The groups of related accounts listed by the query.
  """
    nextPageToken = _messages.StringField(1)
    relatedAccountGroups = _messages.MessageField('GoogleCloudRecaptchaenterpriseV1RelatedAccountGroup', 2, repeated=True)