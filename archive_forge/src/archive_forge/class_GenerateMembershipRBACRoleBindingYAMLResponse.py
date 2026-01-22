from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GenerateMembershipRBACRoleBindingYAMLResponse(_messages.Message):
    """Response for GenerateRBACRoleBindingYAML.

  Fields:
    roleBindingsYaml: a yaml text blob including the RBAC policies.
  """
    roleBindingsYaml = _messages.StringField(1)