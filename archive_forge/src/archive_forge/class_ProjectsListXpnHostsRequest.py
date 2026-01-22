from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ProjectsListXpnHostsRequest(_messages.Message):
    """A ProjectsListXpnHostsRequest object.

  Fields:
    organization: Optional organization ID managed by Cloud Resource Manager,
      for which to list shared VPC host projects. If not specified, the
      organization will be inferred from the project.
  """
    organization = _messages.StringField(1)