from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsKeyvaluemapsDeleteRequest(_messages.Message):
    """A ApigeeOrganizationsKeyvaluemapsDeleteRequest object.

  Fields:
    name: Required. Name of the key value map. Use the following structure in
      your request: `organizations/{org}/keyvaluemaps/{keyvaluemap}`
  """
    name = _messages.StringField(1, required=True)