from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareAdminManualLbConfig(_messages.Message):
    """A VmwareAdminManualLbConfig object.

  Fields:
    addonsNodePort: NodePort for add-ons server in the admin cluster.
    controlPlaneNodePort: NodePort for control plane service. The Kubernetes
      API server in the admin cluster is implemented as a Service of type
      NodePort (ex. 30968).
    ingressHttpNodePort: NodePort for ingress service's http. The ingress
      service in the admin cluster is implemented as a Service of type
      NodePort (ex. 32527).
    ingressHttpsNodePort: NodePort for ingress service's https. The ingress
      service in the admin cluster is implemented as a Service of type
      NodePort (ex. 30139).
    konnectivityServerNodePort: NodePort for konnectivity server service
      running as a sidecar in each kube-apiserver pod (ex. 30564).
  """
    addonsNodePort = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    controlPlaneNodePort = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    ingressHttpNodePort = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    ingressHttpsNodePort = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    konnectivityServerNodePort = _messages.IntegerField(5, variant=_messages.Variant.INT32)