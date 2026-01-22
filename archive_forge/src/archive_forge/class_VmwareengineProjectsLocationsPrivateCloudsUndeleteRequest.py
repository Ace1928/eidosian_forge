from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmwareengineProjectsLocationsPrivateCloudsUndeleteRequest(_messages.Message):
    """A VmwareengineProjectsLocationsPrivateCloudsUndeleteRequest object.

  Fields:
    name: Required. The resource name of the private cloud scheduled for
      deletion. Resource names are schemeless URIs that follow the conventions
      in https://cloud.google.com/apis/design/resource_names. For example:
      `projects/my-project/locations/us-central1-a/privateClouds/my-cloud`
    undeletePrivateCloudRequest: A UndeletePrivateCloudRequest resource to be
      passed as the request body.
  """
    name = _messages.StringField(1, required=True)
    undeletePrivateCloudRequest = _messages.MessageField('UndeletePrivateCloudRequest', 2)