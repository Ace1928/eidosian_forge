from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsApiproductsAttributesRequest(_messages.Message):
    """A ApigeeOrganizationsApiproductsAttributesRequest object.

  Fields:
    googleCloudApigeeV1Attributes: A GoogleCloudApigeeV1Attributes resource to
      be passed as the request body.
    name: Required. Name of the API product. Use the following structure in
      your request: `organizations/{org}/apiproducts/{apiproduct}`
  """
    googleCloudApigeeV1Attributes = _messages.MessageField('GoogleCloudApigeeV1Attributes', 1)
    name = _messages.StringField(2, required=True)