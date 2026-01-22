from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DatafusionProjectsLocationsInstancesRestartRequest(_messages.Message):
    """A DatafusionProjectsLocationsInstancesRestartRequest object.

  Fields:
    name: Required. Name of the Data Fusion instance which need to be
      restarted in the form of
      projects/{project}/locations/{location}/instances/{instance}
    restartInstanceRequest: A RestartInstanceRequest resource to be passed as
      the request body.
  """
    name = _messages.StringField(1, required=True)
    restartInstanceRequest = _messages.MessageField('RestartInstanceRequest', 2)