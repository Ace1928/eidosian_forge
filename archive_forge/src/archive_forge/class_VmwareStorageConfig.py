from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareStorageConfig(_messages.Message):
    """Specifies vSphere CSI components deployment config in the VMware user
  cluster.

  Fields:
    vsphereCsiDisabled: Whether or not to deploy vSphere CSI components in the
      VMware user cluster. Enabled by default.
  """
    vsphereCsiDisabled = _messages.BooleanField(1)