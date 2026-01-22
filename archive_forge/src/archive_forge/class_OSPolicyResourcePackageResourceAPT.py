from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OSPolicyResourcePackageResourceAPT(_messages.Message):
    """A package managed by APT. - install: `apt-get update && apt-get -y
  install [name]` - remove: `apt-get -y remove [name]`

  Fields:
    name: Required. Package name.
  """
    name = _messages.StringField(1)