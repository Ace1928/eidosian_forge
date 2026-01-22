from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesUpgradeInternalRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesUpgradeInternalRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    upgradeInstanceInternalRequest: A UpgradeInstanceInternalRequest resource
      to be passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    upgradeInstanceInternalRequest = _messages.MessageField('UpgradeInstanceInternalRequest', 2)