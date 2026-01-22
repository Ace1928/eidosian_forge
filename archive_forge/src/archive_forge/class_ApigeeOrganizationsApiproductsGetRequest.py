from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApiproductsGetRequest(_messages.Message):
    """A ApigeeOrganizationsApiproductsGetRequest object.

  Fields:
    name: Required. Name of the API product. Use the following structure in
      your request: `organizations/{org}/apiproducts/{apiproduct}`
  """
    name = _messages.StringField(1, required=True)