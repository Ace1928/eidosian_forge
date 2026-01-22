from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsDevelopersAttributesListRequest(_messages.Message):
    """A ApigeeOrganizationsDevelopersAttributesListRequest object.

  Fields:
    parent: Required. Email address of the developer for which attributes are
      being listed. Use the following structure in your request:
      `organizations/{org}/developers/{developer_email}`
  """
    parent = _messages.StringField(1, required=True)