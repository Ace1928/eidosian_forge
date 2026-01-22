from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AuthztoolkitProjectsLocationsPoliciesListRequest(_messages.Message):
    """A AuthztoolkitProjectsLocationsPoliciesListRequest object.

  Fields:
    filter: Query filter. Filters must adhere to the following rules: *
      Boolean values must be unquoted "true" or "false" string literals. *
      String values must be double-quoted. * Wildcard character("*") is
      limited to use with the has operator (":"), and can be used only at the
      end of a string literal. * Timestamps must be quoted strings in the
      RFC3339 format. Example : filter=create_time>"2022-05-09T22:28:28Z"
      Filters support logical operators - AND, OR, NOT (Note: OR has higher
      precedence than AND)
    orderBy: Criteria for ordering results. Currently supported fields for
      ordering - name and create_time. Example: order_by="name
      desc,create_time desc".
    pageSize: Requested page size. Server may return fewer items than
      requested. The maximum allowed value is 50, values above this will be
      coerced to 50. Default value: 50
    pageToken: Next page token, received from a previous policies.list call.
      When paginating, all other input parameters (except page_token) provided
      to policies.list call must remain the same.
    parent: Required. Parent value for ListPoliciesRequest
  """
    filter = _messages.StringField(1)
    orderBy = _messages.StringField(2)
    pageSize = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(4)
    parent = _messages.StringField(5, required=True)