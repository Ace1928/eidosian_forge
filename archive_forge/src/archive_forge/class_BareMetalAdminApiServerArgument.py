from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BareMetalAdminApiServerArgument(_messages.Message):
    """BareMetalAdminApiServerArgument represents an arg name->value pair. Only
  a subset of customized flags are supported. Please refer to the API server
  documentation below to know the exact format:
  https://kubernetes.io/docs/reference/command-line-tools-reference/kube-
  apiserver/

  Fields:
    argument: Required. The argument name as it appears on the API Server
      command line please make sure to remove the leading dashes.
    value: Required. The value of the arg as it will be passed to the API
      Server command line.
  """
    argument = _messages.StringField(1)
    value = _messages.StringField(2)