from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AttackPathNode(_messages.Message):
    """Represents one point that an attacker passes through in this attack
  path.

  Fields:
    associatedFindings: The findings associated with this node in the attack
      path.
    attackSteps: A list of attack step nodes that exist in this attack path
      node.
    displayName: Human-readable name of this resource.
    resource: The name of the resource at this point in the attack path. The
      format of the name follows the Cloud Asset Inventory [resource name
      format]("https://cloud.google.com/asset-inventory/docs/resource-name-
      format")
    resourceType: The [supported resource
      type](https://cloud.google.com/asset-inventory/docs/supported-asset-
      types")
    uuid: Unique id of the attack path node.
  """
    associatedFindings = _messages.MessageField('PathNodeAssociatedFinding', 1, repeated=True)
    attackSteps = _messages.MessageField('AttackStepNode', 2, repeated=True)
    displayName = _messages.StringField(3)
    resource = _messages.StringField(4)
    resourceType = _messages.StringField(5)
    uuid = _messages.StringField(6)