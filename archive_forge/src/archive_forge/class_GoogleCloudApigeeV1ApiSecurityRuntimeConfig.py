from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1ApiSecurityRuntimeConfig(_messages.Message):
    """Response for
  GetApiSecurityRuntimeConfig[EnvironmentService.GetApiSecurityRuntimeConfig].

  Fields:
    location: A list of up to 5 Cloud Storage Blobs that contain
      SecurityActions.
    name: Name of the environment API Security Runtime configuration resource.
      Format:
      `organizations/{org}/environments/{env}/apiSecurityRuntimeConfig`
    revisionId: Revision ID of the API Security Runtime configuration. The
      higher the value, the more recently the configuration was deployed.
    uid: Unique ID for the API Security Runtime configuration. The ID will
      only change if the environment is deleted and recreated.
    updateTime: Time that the API Security Runtime configuration was updated.
  """
    location = _messages.StringField(1, repeated=True)
    name = _messages.StringField(2)
    revisionId = _messages.IntegerField(3)
    uid = _messages.StringField(4)
    updateTime = _messages.StringField(5)