from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1SecurityProfileEnvironmentAssociation(_messages.Message):
    """Represents a SecurityProfileEnvironmentAssociation resource.

  Fields:
    attachTime: Output only. The time when environment was attached to the
      security profile.
    name: Immutable. Name of the environment that the profile is attached to.
    securityProfileRevisionId: DEPRECATED: DO NOT USE Revision ID of the
      security profile.
  """
    attachTime = _messages.StringField(1)
    name = _messages.StringField(2)
    securityProfileRevisionId = _messages.IntegerField(3)