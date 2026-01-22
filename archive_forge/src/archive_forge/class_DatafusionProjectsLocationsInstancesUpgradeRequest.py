from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatafusionProjectsLocationsInstancesUpgradeRequest(_messages.Message):
    """A DatafusionProjectsLocationsInstancesUpgradeRequest object.

  Fields:
    name: Required. Name of the Data Fusion instance which need to be upgraded
      in the form of
      projects/{project}/locations/{location}/instances/{instance} Instance
      will be upgraded with the latest stable version of the Data Fusion.
    upgradeInstanceRequest: A UpgradeInstanceRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    upgradeInstanceRequest = _messages.MessageField('UpgradeInstanceRequest', 2)