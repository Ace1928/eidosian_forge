from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourcePackageResourceZypper(_messages.Message):
    """A package managed by Zypper. - install: `zypper -y install package` -
  remove: `zypper -y rm package`

  Fields:
    name: Required. Package name.
  """
    name = _messages.StringField(1)