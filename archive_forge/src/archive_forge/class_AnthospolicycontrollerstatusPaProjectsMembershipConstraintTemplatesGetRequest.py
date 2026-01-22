from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AnthospolicycontrollerstatusPaProjectsMembershipConstraintTemplatesGetRequest(_messages.Message):
    """A AnthospolicycontrollerstatusPaProjectsMembershipConstraintTemplatesGet
  Request object.

  Fields:
    name: Required. The name of the membership constraint template to
      retrieve. Format: projects/{project_id}/membershipConstraintTemplates/{c
      onstraint_template_name}/{membership_uuid}.
  """
    name = _messages.StringField(1, required=True)