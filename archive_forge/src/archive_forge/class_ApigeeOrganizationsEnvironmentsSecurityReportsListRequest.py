from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsSecurityReportsListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsSecurityReportsListRequest object.

  Fields:
    dataset: Filter response list by dataset. Example: `api`, `mint`
    from_: Filter response list by returning security reports that created
      after this date time. Time must be in ISO date-time format like
      '2011-12-03T10:15:30Z'.
    pageSize: The maximum number of security report to return in the list
      response.
    pageToken: Token returned from the previous list response to fetch the
      next page.
    parent: Required. The parent resource name. Must be of the form
      `organizations/{org}/environments/{env}`.
    status: Filter response list by security reports status.
    submittedBy: Filter response list by user who submitted queries.
    to: Filter response list by returning security reports that created before
      this date time. Time must be in ISO date-time format like
      '2011-12-03T10:16:30Z'.
  """
    dataset = _messages.StringField(1)
    from_ = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)
    status = _messages.StringField(6)
    submittedBy = _messages.StringField(7)
    to = _messages.StringField(8)