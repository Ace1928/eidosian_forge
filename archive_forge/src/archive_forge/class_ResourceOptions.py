from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceOptions(_messages.Message):
    """ResourceOptions represent options for Kubernetes resource generation.

  Fields:
    connectVersion: Optional. The Connect agent version to use for
      connect_resources. Defaults to the latest GKE Connect version. The
      version must be a currently supported version, obsolete versions will be
      rejected.
    k8sVersion: Optional. Major version of the Kubernetes cluster. This is
      only used to determine which version to use for the
      CustomResourceDefinition resources, `apiextensions/v1beta1`
      or`apiextensions/v1`.
    v1beta1Crd: Optional. Use `apiextensions/v1beta1` instead of
      `apiextensions/v1` for CustomResourceDefinition resources. This option
      should be set for clusters with Kubernetes apiserver versions <1.16.
  """
    connectVersion = _messages.StringField(1)
    k8sVersion = _messages.StringField(2)
    v1beta1Crd = _messages.BooleanField(3)