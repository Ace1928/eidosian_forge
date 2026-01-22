from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsRuntimesUpgradeRequest(_messages.Message):
    """A NotebooksProjectsLocationsRuntimesUpgradeRequest object.

  Fields:
    name: Required. Format:
      `projects/{project_id}/locations/{location}/runtimes/{runtime_id}`
    upgradeRuntimeRequest: A UpgradeRuntimeRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    upgradeRuntimeRequest = _messages.MessageField('UpgradeRuntimeRequest', 2)