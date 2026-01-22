from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalCustomersGetRequest(_messages.Message):
    """A SasportalCustomersGetRequest object.

  Fields:
    name: Required. The name of the customer.
  """
    name = _messages.StringField(1, required=True)