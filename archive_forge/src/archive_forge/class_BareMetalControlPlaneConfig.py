from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalControlPlaneConfig(_messages.Message):
    """Specifies the control plane configuration.

  Fields:
    apiServerArgs: Customizes the default API server args. Only a subset of
      customized flags are supported. For the exact format, refer to the [API
      server documentation](https://kubernetes.io/docs/reference/command-line-
      tools-reference/kube-apiserver/).
    controlPlaneNodePoolConfig: Required. Configures the node pool running the
      control plane.
  """
    apiServerArgs = _messages.MessageField('BareMetalApiServerArgument', 1, repeated=True)
    controlPlaneNodePoolConfig = _messages.MessageField('BareMetalControlPlaneNodePoolConfig', 2)