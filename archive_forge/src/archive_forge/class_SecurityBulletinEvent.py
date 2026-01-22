from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SecurityBulletinEvent(_messages.Message):
    """SecurityBulletinEvent is a notification sent to customers when a
  security bulletin has been posted that they are vulnerable to.

  Fields:
    affectedSupportedMinors: The GKE minor versions affected by this
      vulnerability.
    briefDescription: A brief description of the bulletin. See the bulletin
      pointed to by the bulletin_uri field for an expanded description.
    bulletinId: The ID of the bulletin corresponding to the vulnerability.
    bulletinUri: The URI link to the bulletin on the website for more
      information.
    cveIds: The CVEs associated with this bulletin.
    manualStepsRequired: If this field is specified, it means there are manual
      steps that the user must take to make their clusters safe.
    patchedVersions: The GKE versions where this vulnerability is patched.
    resourceTypeAffected: The resource type (node/control plane) that has the
      vulnerability. Multiple notifications (1 notification per resource type)
      will be sent for a vulnerability that affects > 1 resource type.
    severity: The severity of this bulletin as it relates to GKE.
    suggestedUpgradeTarget: This represents a version selected from the
      patched_versions field that the cluster receiving this notification
      should most likely want to upgrade to based on its current version. Note
      that if this notification is being received by a given cluster, it means
      that this version is currently available as an upgrade target in that
      cluster's location.
  """
    affectedSupportedMinors = _messages.StringField(1, repeated=True)
    briefDescription = _messages.StringField(2)
    bulletinId = _messages.StringField(3)
    bulletinUri = _messages.StringField(4)
    cveIds = _messages.StringField(5, repeated=True)
    manualStepsRequired = _messages.BooleanField(6)
    patchedVersions = _messages.StringField(7, repeated=True)
    resourceTypeAffected = _messages.StringField(8)
    severity = _messages.StringField(9)
    suggestedUpgradeTarget = _messages.StringField(10)