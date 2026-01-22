from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminControlPlaneConfig(_messages.Message):
    """BareMetalAdminControlPlaneConfig specifies the control plane
  configuration.

  Fields:
    apiServerArgs: Customizes the default API server args. Only a subset of
      customized flags are supported. Please refer to the API server
      documentation below to know the exact format:
      https://kubernetes.io/docs/reference/command-line-tools-reference/kube-
      apiserver/
    controlPlaneNodePoolConfig: Required. Configures the node pool running the
      control plane. If specified the corresponding NodePool will be created
      for the cluster's control plane. The NodePool will have the same name
      and namespace as the cluster.
  """
    apiServerArgs = _messages.MessageField('BareMetalAdminApiServerArgument', 1, repeated=True)
    controlPlaneNodePoolConfig = _messages.MessageField('BareMetalAdminControlPlaneNodePoolConfig', 2)