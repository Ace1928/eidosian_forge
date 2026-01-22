from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterUpgradeScopeSpec(_messages.Message):
    """**ClusterUpgrade**: The configuration for the scope-level ClusterUpgrade
  feature.

  Fields:
    gkeUpgradeOverrides: Allow users to override some properties of each GKE
      upgrade.
    postConditions: Required. Post conditions to evaluate to mark an upgrade
      COMPLETE. Required.
    upstreamScopes: This scope consumes upgrades that have COMPLETE status
      code in the upstream scopes. See UpgradeStatus.Code for code
      definitions. The scope name should be in the form:
      `projects/{p}/locations/global/scopes/{s}` Where {p} is the project, {s}
      is a valid Scope in this project. {p} WILL match the Feature's project.
      This is defined as repeated for future proof reasons. Initial
      implementation will enforce at most one upstream scope.
  """
    gkeUpgradeOverrides = _messages.MessageField('ClusterUpgradeGKEUpgradeOverride', 1, repeated=True)
    postConditions = _messages.MessageField('ClusterUpgradePostConditions', 2)
    upstreamScopes = _messages.StringField(3, repeated=True)