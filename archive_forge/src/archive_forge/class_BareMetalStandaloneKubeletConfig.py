from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalStandaloneKubeletConfig(_messages.Message):
    """KubeletConfig defines the modifiable kubelet configurations for
  baremetal machines. Note: this list includes fields supported in GKE (see
  https://cloud.google.com/kubernetes-engine/docs/how-to/node-system-
  config#kubelet-options).

  Fields:
    registryBurst: The maximum size of bursty pulls, temporarily allows pulls
      to burst to this number, while still not exceeding registry_pull_qps.
      The value must not be a negative number. Updating this field may impact
      scalability by changing the amount of traffic produced by image pulls.
      Defaults to 10.
    registryPullQps: The limit of registry pulls per second. Setting this
      value to 0 means no limit. Updating this field may impact scalability by
      changing the amount of traffic produced by image pulls. Defaults to 5.
    serializeImagePullsDisabled: Prevents the Kubelet from pulling multiple
      images at a time. We recommend *not* changing the default value on nodes
      that run docker daemon with version < 1.9 or an Another Union File
      System (Aufs) storage backend. Issue
      https://github.com/kubernetes/kubernetes/issues/10959 has more details.
  """
    registryBurst = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    registryPullQps = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    serializeImagePullsDisabled = _messages.BooleanField(3)