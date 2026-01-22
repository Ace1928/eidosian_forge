from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NamespaceValueValuesEnum(_messages.Enum):
    """Required. The namespace that the credential belongs to.

    Values:
      NAMESPACE_UNSPECIFIED: The default namespace.
      GITHUB_ENTERPRISE: A credential to be used with GitHub enterprise.
    """
    NAMESPACE_UNSPECIFIED = 0
    GITHUB_ENTERPRISE = 1