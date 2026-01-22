from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SasportalCustomersNodesCreateRequest(_messages.Message):
    """A SasportalCustomersNodesCreateRequest object.

  Fields:
    parent: Required. The parent resource name where the node is to be
      created.
    sasPortalNode: A SasPortalNode resource to be passed as the request body.
  """
    parent = _messages.StringField(1, required=True)
    sasPortalNode = _messages.MessageField('SasPortalNode', 2)