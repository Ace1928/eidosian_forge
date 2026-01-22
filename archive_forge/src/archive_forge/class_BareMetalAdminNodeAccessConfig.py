from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminNodeAccessConfig(_messages.Message):
    """Specifies the node access related settings for the bare metal admin
  cluster.

  Fields:
    loginUser: Required. LoginUser is the user name used to access node
      machines. It defaults to "root" if not set.
  """
    loginUser = _messages.StringField(1)