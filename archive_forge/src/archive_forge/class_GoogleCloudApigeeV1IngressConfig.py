from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1IngressConfig(_messages.Message):
    """A GoogleCloudApigeeV1IngressConfig object.

  Fields:
    environmentGroups: List of environment groups in the organization.
    name: Name of the resource in the following format:
      `organizations/{org}/deployedIngressConfig`.
    revisionCreateTime: Time at which the IngressConfig revision was created.
    revisionId: Revision id that defines the ordering on IngressConfig
      resources. The higher the revision, the more recently the configuration
      was deployed.
    uid: A unique id for the ingress config that will only change if the
      organization is deleted and recreated.
  """
    environmentGroups = _messages.MessageField('GoogleCloudApigeeV1EnvironmentGroupConfig', 1, repeated=True)
    name = _messages.StringField(2)
    revisionCreateTime = _messages.StringField(3)
    revisionId = _messages.IntegerField(4)
    uid = _messages.StringField(5)