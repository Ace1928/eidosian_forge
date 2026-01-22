from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateSupportBundleRequest(_messages.Message):
    """Request message for SoftwareDefinedDataCenterCLH.GenerateSupportBundle.

  Fields:
    allEsxiNodes: Required. If True, generate the support bundle of all ESXi
      nodes.
    esxiNodeNames: Optional. If all_esxi_nodes is not true, generate the
      support bundle of specified ESXi nodes.
    nsxt: Required. If True, generate the support bundle of NSX-T.
    vcenter: Required. If True, generate the support bundle of vCenter.
  """
    allEsxiNodes = _messages.BooleanField(1)
    esxiNodeNames = _messages.StringField(2, repeated=True)
    nsxt = _messages.BooleanField(3)
    vcenter = _messages.BooleanField(4)