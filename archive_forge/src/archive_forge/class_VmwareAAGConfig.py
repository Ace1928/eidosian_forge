from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAAGConfig(_messages.Message):
    """Specifies anti affinity group config for the VMware user cluster.

  Fields:
    aagConfigDisabled: Spread nodes across at least three physical hosts
      (requires at least three hosts). Enabled by default.
  """
    aagConfigDisabled = _messages.BooleanField(1)