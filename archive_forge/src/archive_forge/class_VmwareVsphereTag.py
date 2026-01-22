from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareVsphereTag(_messages.Message):
    """VmwareVsphereTag describes a vSphere tag to be placed on VMs in the node
  pool. For more information, see https://docs.vmware.com/en/VMware-
  vSphere/7.0/com.vmware.vsphere.vcenterhost.doc/GUID-E8E854DD-
  AA97-4E0C-8419-CE84F93C4058.html

  Fields:
    category: The Vsphere tag category.
    tag: The Vsphere tag name.
  """
    category = _messages.StringField(1)
    tag = _messages.StringField(2)