from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIamV3betaPolicy(_messages.Message):
    """One of the policies supported by IAM V3

  Fields:
    principalAccessBoundaryPolicy: The principal access boundary kind policy
  """
    principalAccessBoundaryPolicy = _messages.MessageField('GoogleIamV3betaPrincipalAccessBoundaryPolicy', 1)