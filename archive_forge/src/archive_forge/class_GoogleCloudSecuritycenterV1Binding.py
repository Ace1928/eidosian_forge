from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1Binding(_messages.Message):
    """Represents a Kubernetes RoleBinding or ClusterRoleBinding.

  Fields:
    name: Name for the binding.
    ns: Namespace for the binding.
    role: The Role or ClusterRole referenced by the binding.
    subjects: Represents one or more subjects that are bound to the role. Not
      always available for PATCH requests.
  """
    name = _messages.StringField(1)
    ns = _messages.StringField(2)
    role = _messages.MessageField('Role', 3)
    subjects = _messages.MessageField('Subject', 4, repeated=True)