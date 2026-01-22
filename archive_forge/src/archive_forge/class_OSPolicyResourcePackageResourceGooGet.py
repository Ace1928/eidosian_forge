from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourcePackageResourceGooGet(_messages.Message):
    """A package managed by GooGet. - install: `googet -noconfirm install
  package` - remove: `googet -noconfirm remove package`

  Fields:
    name: Required. Package name.
  """
    name = _messages.StringField(1)