from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV2AwsOrganizationalUnit(_messages.Message):
    """An Organizational Unit (OU) is a container of AWS accounts within a root
  of an organization. Policies that are attached to an OU apply to all
  accounts contained in that OU and in any child OUs.

  Fields:
    id: The unique identifier (ID) associated with this OU. The regex pattern
      for an organizational unit ID string requires "ou-" followed by from 4
      to 32 lowercase letters or digits (the ID of the root that contains the
      OU). This string is followed by a second "-" dash and from 8 to 32
      additional lowercase letters or digits. For example, "ou-ab12-cd34ef56".
    name: The friendly name of the OU.
  """
    id = _messages.StringField(1)
    name = _messages.StringField(2)