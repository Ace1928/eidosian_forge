from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DomainsProjectsLocationsRegistrationsListRequest(_messages.Message):
    """A DomainsProjectsLocationsRegistrationsListRequest object.

  Fields:
    filter: Filter expression to restrict the `Registration`s returned. The
      expression must specify the field name, a comparison operator, and the
      value that you want to use for filtering. The value must be a string, a
      number, a boolean, or an enum value. The comparison operator should be
      one of =, !=, >, <, >=, <=, or : for prefix or wildcard matches. For
      example, to filter to a specific domain name, use an expression like
      `domainName="example.com"`. You can also check for the existence of a
      field; for example, to find domains using custom DNS settings, use an
      expression like `dnsSettings.customDns:*`. You can also create compound
      filters by combining expressions with the `AND` and `OR` operators. For
      example, to find domains that are suspended or have specific issues
      flagged, use an expression like `(state=SUSPENDED) OR (issue:*)`.
    pageSize: Maximum number of results to return.
    pageToken: When set to the `next_page_token` from a prior response,
      provides the next page of results.
    parent: Required. The project and location from which to list
      `Registration`s, specified in the format `projects/*/locations/*`.
  """
    filter = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)