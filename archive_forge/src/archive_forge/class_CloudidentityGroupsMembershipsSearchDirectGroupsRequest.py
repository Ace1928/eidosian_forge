from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudidentityGroupsMembershipsSearchDirectGroupsRequest(_messages.Message):
    """A CloudidentityGroupsMembershipsSearchDirectGroupsRequest object.

  Fields:
    orderBy: The ordering of membership relation for the display name or email
      in the response. The syntax for this field can be found at
      https://cloud.google.com/apis/design/design_patterns#sorting_order.
      Example: Sort by the ascending display name: order_by="group_name" or
      order_by="group_name asc". Sort by the descending display name:
      order_by="group_name desc". Sort by the ascending group key:
      order_by="group_key" or order_by="group_key asc". Sort by the descending
      group key: order_by="group_key desc".
    pageSize: The default page size is 200 (max 1000).
    pageToken: The next_page_token value returned from a previous list
      request, if any
    parent: [Resource
      name](https://cloud.google.com/apis/design/resource_names) of the group
      to search transitive memberships in. Format: groups/{group_id}, where
      group_id is always '-' as this API will search across all groups for a
      given member.
    query: Required. A CEL expression that MUST include member specification
      AND label(s). Users can search on label attributes of groups. CONTAINS
      match ('in') is supported on labels. Identity-mapped groups are uniquely
      identified by both a `member_key_id` and a `member_key_namespace`, which
      requires an additional query input: `member_key_namespace`. Example
      query: `member_key_id == 'member_key_id_value' && 'label_value' in
      labels`
  """
    orderBy = _messages.StringField(1)
    pageSize = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(3)
    parent = _messages.StringField(4, required=True)
    query = _messages.StringField(5)