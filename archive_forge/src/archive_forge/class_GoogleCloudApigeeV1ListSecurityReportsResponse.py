from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ListSecurityReportsResponse(_messages.Message):
    """The response for SecurityReports.

  Fields:
    nextPageToken: If the number of security reports exceeded the page size
      requested, the token can be used to fetch the next page in a subsequent
      call. If the response is the last page and there are no more reports to
      return this field is left empty.
    securityReports: The security reports belong to requested resource name.
  """
    nextPageToken = _messages.StringField(1)
    securityReports = _messages.MessageField('GoogleCloudApigeeV1SecurityReport', 2, repeated=True)